#ifndef EXTACT_RELATED_H_
#define EXTACT_RELATED_H_

#include "command_line_parameters_defs.h"
#include "sim_estimator.h"

#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <thread>
#include <filesystem>
#include <regex>
#include <numeric>


// Извлечение из модели словарей пар связанных слов
class RelatedPairsExtractor
{
public:
  // извлечение
  void run( const CommandLineParametersDefs& cmdLineParams )
  {
    // 1. Создаем оценщик близости
    if ( !cmdLineParams.isDefined("-model") )
    {
      std::cerr << "-model parameter must be defined." << std::endl;
      return;
    }
    sim_estimator = std::make_shared<SimilarityEstimator>( cmdLineParams.getAsFloat("-a_ratio") );
    if ( !sim_estimator->load_model(cmdLineParams.getAsString("-model")) )
    {
      return;
    }

    // 2. Открываем файлы для сохранения результатов
    if ( !cmdLineParams.isDefined("-rr_vocab") )
    {
      std::cerr << "-rr_vocab parameters must be defined." << std::endl;
      return;
    }
    std::ofstream ofs_dep(cmdLineParams.getAsString("-rr_vocab")+".dep");
    if ( !ofs_dep.good() )
      return;
    std::ofstream ofs_assoc(cmdLineParams.getAsString("-rr_vocab")+".assoc");
    if ( !ofs_assoc.good() )
      return;
    dep_ofs = &ofs_dep;
    assoc_ofs = &ofs_assoc;

    // 3. Создаем рабочие потоки для вычисления близости и сохраннения результатов
    min_sim = cmdLineParams.getAsFloat("-rr_min_sim");
    size_t threads_count = cmdLineParams.getAsInt("-threads");
    std::vector<std::thread> threads_vec;
    threads_vec.reserve(threads_count);
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec.emplace_back(&RelatedPairsExtractor::thread_func, this);
    // ждем завершения потоков
    for (size_t i = 0; i < threads_count; ++i)
      threads_vec[i].join();

    sim_estimator.reset();
    std::cout << std::endl;
  } // method-end
  // мержинг извлеченного из нескольких моделей
  void merge( const CommandLineParametersDefs& cmdLineParams )
  {
    if ( !cmdLineParams.isDefined("-rr_vocab") ) // здесь содержится имя каталога, где лежат извлеченные из моделей данные
    {
      std::cerr << "-rr_vocab parameters must be defined." << std::endl;
      return;
    }
    min_sim = cmdLineParams.getAsFloat("-rr_min_sim");

    // 1. считываем данные из файлов
    std::cout << "Read data from files" << std::endl;
    std::map<std::string, std::map<std::string, std::vector<float>>> dep_pairs, assoc_pairs;
    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    for ( const auto& entry : recursive_directory_iterator( cmdLineParams.getAsString("-rr_vocab") ) )
    {
      if ( !entry.is_regular_file() ) continue;
      if( entry.path().extension() == ".dep" )
      {
        std::cout << entry.path().string() << std::endl;
        read_to_merge(dep_pairs, entry.path().string());
      }
      if( entry.path().extension() == ".assoc" )
      {
        std::cout << entry.path().string() << std::endl;
        read_to_merge(assoc_pairs, entry.path().string());
      }
    }
    std::cout << "Total. Dep: " << dep_pairs.size() << ", Assoc: " << assoc_pairs.size() << std::endl;

    // 2. строим сводный файл
    std::ofstream ofs_dep_result("r.dep");
    std::ofstream ofs_assoc_result("r.assoc");
    if ( !ofs_dep_result.good() || !ofs_assoc_result.good() )
    {
      std::cerr << "can't create resulting files" << std::endl;
      return;
    }
    for ( const auto& [w1, m] : dep_pairs )
    {
      std::multimap<float, std::string, std::greater<float>> avg;
      for ( const auto& [w2, v] : m )
      {
        if ( v.size() < 3 ) continue; // фильтруем ненадежное
        const double avg_v = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        avg.insert( std::make_pair(avg_v, w2) );
      }
      size_t rec_cnt = 0;
      for (const auto& [v, w2] : avg )
      {
        if (rec_cnt++ > 10) break;
        ofs_dep_result << w1 << " " << w2 << " " << v << std::endl;
      }
    }
    for ( const auto& [w1, m] : assoc_pairs )
    {
      std::multimap<float, std::string, std::greater<float>> avg;
      for ( const auto& [w2, v] : m )
      {
        if ( v.size() < 3 ) continue; // фильтруем ненадежное
        const double avg_v = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        avg.insert( std::make_pair(avg_v, w2) );
      }
      size_t rec_cnt = 0;
      for (const auto& [v, w2] : avg )
      {
        if (rec_cnt++ > 10) break;
        ofs_assoc_result << w1 << " " << w2 << " " << v << std::endl;
      }
    }

  } // method-end
private:
  int widx = -1;
  std::mutex mtx_wi;
  std::mutex mtx_out;
  std::ofstream* dep_ofs;
  std::ofstream* assoc_ofs;
  float min_sim;
  std::shared_ptr<SimilarityEstimator> sim_estimator;

  size_t get_next_word_index()
  {
    std::lock_guard<std::mutex> lock(mtx_wi);
    ++widx;
    std::cout << widx << '\r';
    std::cout.flush();
    return widx;
  }

  void save(std::ofstream* ofs, const std::string& s)
  {
    std::lock_guard<std::mutex> lock(mtx_out);
    (*ofs) << s;
  }

  void thread_func()
  {
    auto contains_ru_letter = [](const std::string& lemma) -> bool
        {
          const std::u32string RuLets = U"абвгдеёжзийклмнопрстуфхцчшщьыъэюя";
          auto s32 = StrConv::To_UTF32(lemma);
          return ( s32.find_first_of(RuLets) != std::u32string::npos );
        };

    auto try_get = [](std::multimap<float, std::string, std::greater<float>>& best, float sim, const std::string& word)
        {
          if (best.size() < 50)
            best.insert( std::make_pair(sim, word) );
          else
          {
            auto minIt = std::prev( best.end() );
            if (sim > minIt->first)
            {
              best.erase(minIt);
              best.insert( std::make_pair(sim, word) );
            }
          }
        };

    auto to_str = [](const std::string& word, const std::multimap<float, std::string, std::greater<float>>& best)
        {
          std::string result;
          for (const auto&[sim,w2] : best)
            result += word + " " + w2 + " " + std::to_string(sim) + "\n";
          return result;
        };

    auto vm = sim_estimator->raw();
    while(true)
    {
      size_t i = get_next_word_index();
      if ( i >= vm->words_count ) break;
      if ( !contains_ru_letter(vm->vocab[i]) ) continue;
      std::multimap<float, std::string, std::greater<float>> best_dep, best_assoc;
      for (size_t j = 0; j < vm->words_count; ++j)
      {
        if ( i == j ) continue;
        if ( !contains_ru_letter(vm->vocab[j]) ) continue;
        auto dep_sim = sim_estimator->get_sim(SimilarityEstimator::cdDepOnly, i, j);
        if (dep_sim && dep_sim.value() > min_sim)
          try_get(best_dep, dep_sim.value(), vm->vocab[j]);
        auto assoc_sim = sim_estimator->get_sim(SimilarityEstimator::cdAssocOnly, i, j);
        if (assoc_sim && assoc_sim.value() > min_sim)
          try_get(best_assoc, assoc_sim.value(), vm->vocab[j]);
      }
      if ( !best_dep.empty() )
        save(dep_ofs, to_str(vm->vocab[i], best_dep));
      if ( !best_assoc.empty() )
        save(assoc_ofs, to_str(vm->vocab[i], best_assoc));
    }

  }

  void read_to_merge(std::map<std::string, std::map<std::string, std::vector<float>>>& data, const std::string& fn)
  {
    std::ifstream ifs(fn);
    if (!ifs.good())
      return;
    std::string line;
//size_t dbg_cnt=0;
    while ( std::getline(ifs, line).good() )
    {
      const std::regex space_re("\\s+");
      std::vector<std::string> rec_items { std::sregex_token_iterator(line.cbegin(), line.cend(), space_re, -1), std::sregex_token_iterator() };
      if (rec_items.size() != 3)
      {
        std::cerr << "Invalid record in " << fn << ": " << line << std::endl;
        return;
      }
      float val = 0;
      try { val = std::stof(rec_items[2]); }
      catch (...) {
        std::cerr << "Invalid float value in '" << line << "' in " << fn << std::endl;
        continue;
      }
      data[rec_items[0]][rec_items[1]].push_back(val);
//if (dbg_cnt++ > 10000) break;
    }
  }
}; // class-decl-end


#endif /* EXTACT_RELATED_H_ */
