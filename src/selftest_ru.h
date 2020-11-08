#ifndef SELFTEST_RU_H_
#define SELFTEST_RU_H_

#include "sim_estimator.h"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <regex>
#include <optional>
#include <cmath>
#include <iterator>

// процедура оценки качества модели для русского языка (быстрая самодиагностика)
class SelfTest_ru
{
public:
  SelfTest_ru( std::shared_ptr<SimilarityEstimator> sim_estimator, bool replace_yo_in_russe)
  : sim_meter(sim_estimator)
  , russe_replace_yo(replace_yo_in_russe)
  {
  }
  void run(bool verbose = false)
  {
    test_floats(verbose);
    std::cout << std::endl;
    test_nonsim_dep(verbose);
    std::cout << std::endl;
    test_nonsim_assoc(verbose);
    std::cout << std::endl;
    test_sim_dep(verbose);
    std::cout << std::endl;
    test_sim_assoc(verbose);
//    std::cout << std::endl;
//    test_sim_all(verbose);
    std::cout << std::endl;
    dimensions_analyse(verbose);
    std::cout << std::endl;
    test_russe2015();
    std::cout << std::endl;
    test_rusim();
  } // method-end
private:
  // указатель на объект для оценки семантической близости
  std::shared_ptr<SimilarityEstimator> sim_meter;
  // нужно ли замещать букву "ё" в тестах RUSSE
  bool russe_replace_yo;

  // протоколирование в файл
  void log(const std::string& msg) const
  {
    static std::shared_ptr<std::ofstream> log_ofs;
    if (!log_ofs)
      log_ofs = std::make_shared<std::ofstream>("self-test.log");
    (*log_ofs) << msg << std::endl;
  }

  // тест корректности значений в векторах (валидация вещественных чисел)
  void test_floats(bool verbose = false)
  {
    std::cout << "Run float checker" << std::endl;
    size_t vocab_size = 0, emb_size = 0;
    float* embeddings = nullptr;
    std::vector<std::string> vocab;
    sim_meter->raw(vocab_size, emb_size, embeddings, vocab);
    auto float_class_func = [](float x)
        {
            switch (std::fpclassify(x))
            {
                case FP_INFINITE:  return "Inf";
                case FP_NAN:       return "NaN";
                case FP_NORMAL:    return "normal";
                case FP_SUBNORMAL: return "subnormal";
                case FP_ZERO:      return "zero";
                default:           return "unknown";
            }
        };
    bool all_right = true;
    for (size_t w = 0; w < vocab_size; ++w)
      for (size_t d = 0; d < emb_size; ++d)
      {
        if ( !std::isnormal(*(embeddings+w*emb_size+d)) )
        {
          std::cout << "  " << vocab[w] << " has abnormal value. -- " << float_class_func(*(embeddings+w*emb_size+d)) << std::endl;
          all_right = false;
          break;
        }
      }
    if (all_right)
      std::cout << "  All floating point values are correct." << std::endl;
  }

  // тест категориально несвязанных (среднее расстояние между ними должно быть <=0 )
  void test_nonsim_dep(bool verbose = false)
  {
    std::cout << "Run test_nonsim_dep" << std::endl;
    const std::vector< std::pair<std::string, std::string> > TEST_DATA = {
        {"синий", "президент"},
        {"синий", "идея"},
        {"синий", "восемь"},
        {"синий", "он"},
        {"синий", "верх"},
        {"синий", "в"},
        {"синий", "не"},
        {"синий", "бежать"},
        {"синий", "неделя"},
        {"синий", "рубль"},
        {"синий", "китаец"},
        {"синий", "бензин"},
        {"синий", "во-первых"},
        {"синий", "быстро"},
        {"идея", "бежать"},
        {"идея", "в"},
        {"идея", "восемь"},
        {"идея", "кофе"},
        {"идея", "каменный"},
        {"идея", "не"},
        {"идея", "он"},
        {"идея", "быстро"},
        {"идея", "верх"},
        {"восемь", "бежать"},
        {"восемь", "в"},
        {"восемь", "верх"},
        {"восемь", "президент"},
        {"восемь", "быстро"},
        {"верх", "бежать"},
        {"в", "он"},
        {"в", "китаец"},
        {"в", "сообщить"},
        {"в", "быстро"},
        {"в", "измениться"},
        {"в", "секретно"},
        {"у", "быть"},
        {"из", "общеизвестно"},
        {"он", "бежать"},
        {"он", "президент"},
        {"они", "быстро"},
        {"они", "купить"},
        {"они", "мочь"},
        {"они", "американец"},
        {"я", "виснуть"},
        {"математика", "быстро"},
        {"математика", "во-первых"},
        {"математика", "кофе"},
        {"бежать", "кофе"},
        {"бежать", "во-первых"},
        {"бежать", "вторник"},
        {"бежать", "только"},
        {"бежать", "автомобиль"},
        {"молоко", "митинг"},
        {"увлекшийся", "молоко"},
        {"кумыс", "разбежаться"},
        {"раджа", "собираться"},
        {"чашка", "кипятить"},
        {"темно-красный", "кипятить"},
        {"украшенный", "тепловоз"},
        {"забытый", "книга"},
        {"атакованный", "возле"},
        {"дотащиться", "около"},
        {"проснуться", "вельможа"},
        {"пруд", "вдохнуть"},
        {"пулемет", "лодочный"},
        {"вовремя", "рак-отшельник"},
        {"камыш", "виртуоз"},
        {"промокнуть", "престижно"},
        {"сообщить", "лошадь"},
        {"сообщить", "море"}
    };
    float avg_sim = 0, max_sim = -100;
    size_t cnt = 0;
    std::string max_pair;
    for (auto& d : TEST_DATA)
    {
      auto sim = sim_meter->get_sim(SimilarityEstimator::cdDepOnly, d.first, d.second);
      if (!sim)
      {
        std::cout << "  warn: pair not found <" << d.first << ", " << d.second << ">" << std::endl;
        continue;
      }
      avg_sim += sim.value();
      ++cnt;
      if (sim.value() > max_sim)
      {
        max_sim = sim.value();
        max_pair = d.first + ", " + d.second;
      }
      if (verbose)
        std::cout << d.first << ", " << d.second << "\t" << (sim.value()) << std::endl;
    }
    avg_sim /= cnt;
    std::cout << "  AVG = " << avg_sim << "  (the less, the better)" << std::endl;
    std::cout << "  MAX = " << max_sim << " -- " << max_pair << std::endl;
  } // method-end

  // тест ассоциативно несвязанных (среднее расстояние между ними должно быть <=0 )
  void test_nonsim_assoc(bool verbose = false)
  {
    std::cout << "Run test_nonsim_assoc" << std::endl;
    const std::vector< std::pair<std::string, std::string> > TEST_DATA = {
        {"стеклянный", "президент"},
        {"танк", "астероид"},
        {"футбол", "свинья"},
        {"спать", "кран"},
        {"этаж", "тоска"},
        {"картофель", "математика"},
        {"загорать", "балет"},
        {"бинокль", "ботинок"},
        {"сосна", "лейтенант"},
        {"министерство", "мяч"},
        {"футбольный", "причал"},
        {"международный", "свинья"},
        {"облако", "цех"},
        {"зарплата", "материк"},
        {"доллар", "гроза"},
        {"ветер", "мох"},
        {"кабинет", "планета"},
        {"лечить", "балет"},
        {"фабрика", "пляж"},
        {"хлеб", "затея"},
        {"маркиз", "ракета"},
        {"посох", "катер"},
        {"плыть", "миллиметр"},
        {"салат", "вата"},
        {"вкусный", "станок"},
        {"война", "хоккеист"},
        {"лекарь", "смартфон"},
        {"автомобиль", "рыцарь"},
        {"императорский", "океан"},
        {"атомный", "курица"},
        {"записка", "льдина"},
        {"рыба", "кольцо"},
        {"хирург", "истребитель"},
        {"джип", "крокодил"},
        {"ведро", "кредит"},
        {"школьник", "сенат"},
        {"школьный", "указ"},
        {"ходатайство", "луна"},
        {"дверь", "нефть"},
        {"нефтяной", "учитель"},
        {"шахта", "республиканец"},
        {"мост", "препарат"},
        {"бочка", "тетрадь"},
        {"муха", "атомный"},
        {"кожаный", "станция"},
        {"кожаный", "атомный"},
        {"рация", "мюон"},
        {"флаг", "скот"},
        {"паспорт", "орех"},
        {"атом", "город"}
    };
    float avg_sim = 0, max_sim = -100;
    size_t cnt = 0;
    std::string max_pair;
    for (auto& d : TEST_DATA)
    {
      auto sim = sim_meter->get_sim(SimilarityEstimator::cdAssocOnly, d.first, d.second);
      if (!sim)
      {
        std::cout << "  warn: pair not found <" << d.first << ", " << d.second << ">" << std::endl;
        continue;
      }
      avg_sim += sim.value();
      ++cnt;
      if (sim.value() > max_sim)
      {
        max_sim = sim.value();
        max_pair = d.first + ", " + d.second;
      }
      if (verbose)
        std::cout << d.first << ", " << d.second << "\t" << (sim.value()) << std::endl;
    }
    avg_sim /= cnt;
    std::cout << "  AVG = " << avg_sim << "  (the less, the better)" << std::endl;
    std::cout << "  MAX = " << max_sim << " -- " << max_pair << std::endl;
  } // method-end

  // тест категориально связанных (среднее расстояние между ними должно стремиться к 1 )
  void test_sim_dep(bool verbose = false)
  {
    std::cout << "Run test_sim_dep" << std::endl;
    const std::vector< std::pair<std::string, std::string> > TEST_DATA = {
        {"маркиз", "король"},
        {"министр", "король"},
        {"повелитель", "король"},
        {"врач", "лекарь"},
        {"повозка", "телега"},
        {"президент", "лидер"},
        {"математика", "физика"},
        {"корабль", "пароход"},
        {"самолет", "бомбардировщик"},
        {"компьютер", "ноутбук"},
        {"бежать", "идти"},
        {"завод", "фабрика"},
        {"китаец", "мексиканец"},
        {"сказать", "говорить"},
        {"тьма", "мрак"},
        {"лазурный", "фиолетовый"},
        {"быстрый", "скорый"},
        {"гигантский", "крупный"},
        {"вторник", "четверг"},
        {"он", "они"},
        {"восемь", "шесть"},
        {"картофель", "морковь"},
        {"атом", "молекула"},
        {"водород", "кислород"},
        {"недавно", "накануне"},
        {"в", "на"},
        {"теперь", "тогда"},
        {"вверху", "внизу"},
        {"верх", "низ"},
        {"доллар", "динар"},
        {"немецкий", "французский"},
        {"город", "поселок"},
        {"озеро", "река"},
        {"гора", "холм"},
        {"сосна", "береза"},
        {"чай", "сок"},
        {"омлет", "суп"},
        {"купить", "продать"},
        {"смотреть", "наблюдать"},
        {"бочка", "ведро"},
        {"метр", "миллиметр"},
        {"яркий", "цветной"},
        {"стальной", "каменный"},
        {"пуля", "снаряд"},
        {"сержант", "майор"},
        {"лошадь", "кобыла"},
        {"куст", "дерево"},
        {"школа", "вуз"},
        {"учитель", "физрук"},
        {"лечить", "лечение"}
    };
    float avg_sim = 0, min_sim = +100;
    size_t cnt = 0;
    std::string min_pair;
    for (auto& d : TEST_DATA)
    {
      auto sim = sim_meter->get_sim(SimilarityEstimator::cdDepOnly, d.first, d.second);
      if (!sim)
      {
        std::cout << "  warn: pair not found <" << d.first << ", " << d.second << ">" << std::endl;
        continue;
      }
      avg_sim += sim.value();
      ++cnt;
      if (sim.value() < min_sim)
      {
        min_sim = sim.value();
        min_pair = d.first + ", " + d.second;
      }
      if (verbose)
        std::cout << d.first << ", " << d.second << "\t" << (sim.value()) << std::endl;
    }
    avg_sim /= cnt;
    std::cout << "  AVG = " << avg_sim << "  (the more, the better)" << std::endl;
    std::cout << "  MIN = " << min_sim << " -- " << min_pair << std::endl;
  } // method-end

  // тест ассоциативно связанных (среднее расстояние между ними должно стремиться к 1 )
  void test_sim_assoc(bool verbose = false)
  {
    std::cout << "Run test_sim_assoc" << std::endl;
    const std::vector< std::pair<std::string, std::string> > TEST_DATA = {
        {"самолет", "летчик"},
        {"термометр", "температура"},
        {"кастрюля", "суп"},
        {"ягода", "куст"},
        {"пробежать", "марафон"},
        {"лететь", "птица"},
        {"спелый", "яблоко"},
        {"над", "небо"},
        {"лес", "дерево"},
        {"лейтенант", "полиция"},
        {"лейтенант", "армия"},
        {"охотник", "ружье"},
        {"собака", "кличка"},
        {"заводиться", "двигатель"},
        {"автомобиль", "дтп"},
        {"автомобиль", "парковка"},
        {"дождь", "туча"},
        {"погода", "синоптик"},
        {"патриотический", "родина"},
        {"заботливый", "мать"},
        {"стремительно", "мчаться"},
        {"книга", "страница"},
        {"бензин", "заправка"},
        {"корова", "сено"},
        {"клиника", "медсестра"},
        {"король", "трон"},
        {"король", "монархия"},
        {"президент", "брифинг"},
        {"министр", "министерство"},
        {"школа", "учитель"},
        {"школьник", "учебник"},
        {"митинг", "участник"},
        {"выборы", "депутат"},
        {"сенат", "сенатор"},
        {"президент", "президентский"},
        {"школа", "школьный"},
        {"метр", "длина"},
        {"лошадь", "наездник"},
        {"холодильник", "продукт"},
        {"магазин", "товар"},
        {"баррель", "нефть"},
        {"доллар", "стоимость"},
        {"немец", "немецкий"},
        {"море", "берег"},
        {"ветер", "дуть"},
        {"футбол", "нападающий"},
        {"мяч", "ворота"},
        {"шайба", "клюшка"},
        {"шайба", "хоккеист"},
        {"врач", "лечение"}
    };
    float avg_sim = 0, min_sim = +100;
    size_t cnt = 0;
    std::string min_pair;
    for (auto& d : TEST_DATA)
    {
      auto sim = sim_meter->get_sim(SimilarityEstimator::cdAssocOnly, d.first, d.second);
      if (!sim)
      {
        std::cout << "  warn: pair not found <" << d.first << ", " << d.second << ">" << std::endl;
        continue;
      }
      avg_sim += sim.value();
      ++cnt;
      if (sim.value() < min_sim)
      {
        min_sim = sim.value();
        min_pair = d.first + ", " + d.second;
      }
      if (verbose)
        std::cout << d.first << ", " << d.second << "\t" << (sim.value()) << std::endl;
    }
    avg_sim /= cnt;
    std::cout << "  AVG = " << avg_sim << "  (the more, the better)" << std::endl;
    std::cout << "  MIN = " << min_sim << " -- " << min_pair << std::endl;
  } // method-end

  // стат.данные по измерениям
  void dimensions_analyse(bool verbose=false) const
  {
    // получаем доступ к векторному пространству
    size_t vocab_size = 0, emb_size = 0;
    float* embeddings = nullptr;
    std::vector<std::string> vocab;
    sim_meter->raw(vocab_size, emb_size, embeddings, vocab);
    // выводим данные о минимальных и максимальных значениях в пространстве
    const size_t MM_CNT = 5;
    std::map<float, std::pair<std::string,size_t>, std::greater<float>> minValues;
    std::map<float, std::pair<std::string,size_t>> maxValues;
    for (size_t w = 0; w < vocab_size; ++w)
      for (size_t d = 0; d < emb_size; ++d)
      {
        float val = embeddings[w*emb_size+d];
        if (minValues.empty() || val < minValues.begin()->first)
        {
          minValues[val] = std::make_pair(vocab[w],d);
          if (minValues.size() > MM_CNT)
            minValues.erase(minValues.begin());
        }
        if (maxValues.empty() || val > maxValues.begin()->first)
        {
          maxValues[val] = std::make_pair(vocab[w],d);
          if (maxValues.size() > MM_CNT)
            maxValues.erase(maxValues.begin());
        }
      }
    std::cout << "Run dimensions statistic" << std::endl;
    std::cout << "  min-max format: value (word/dimension)" << std::endl;
    std::cout << "    min:";
    for (auto& r : minValues)
      std::cout << "  " << r.first << " (" << r.second.first << "/" << r.second.second << ")";
    std::cout << std::endl;
    std::cout << "    max:";
    for (auto& r : maxValues)
      std::cout << "  " << r.first << " (" << r.second.first << "/" << r.second.second << ")";
    std::cout << std::endl;
    // выводим данные о смещениях нуля в измерениях
    const size_t ZSH_CNT = 5;
    std::map<float, size_t> maxShifts;
    for (size_t d = 0; d < emb_size; ++d)
    {
      float sum = 0.0;
      for (size_t w = 0; w < vocab_size; ++w)
        sum += embeddings[w*emb_size+d];
      sum /= vocab_size;
      sum = fabs(sum);
      if (maxShifts.empty() || sum > maxShifts.begin()->first)
      {
        maxShifts[sum] = d;
        if (maxShifts.size() > ZSH_CNT)
          maxShifts.erase(maxShifts.begin());
      }
    }
    std::cout << "  max zero-shifted dimensions:";
    for (auto& r : maxShifts)
      std::cout << "  " << r.first << " (" << r.second << ")";
    std::cout << std::endl;
    // выводим гистограмму худшего измерения
    std::map<float, size_t> bar;
    const size_t resolution = 100;
    const float step = 1.0 / resolution;
    for (size_t i = 0; i < (2*resolution); ++i)
      bar[-1.0 + i*step] = 0;
    size_t target_dimension = maxShifts.rbegin()->second;
    for (size_t w = 0; w < vocab_size; ++w)
    {
      float val = embeddings[w*emb_size+target_dimension];
      auto it = std::lower_bound(bar.begin(), bar.end(), val, [](const std::pair<float, size_t> item, float bound) {return item.first < bound;});
      if (it != bar.end())
        it->second++;
    }
    auto shrinkLeftFunc = [](size_t cntLim, std::map<float, size_t>& bar)
                          {
                            auto lIt = bar.begin();
                            while (lIt != bar.end() && lIt->second < cntLim)
                              ++lIt;
                            bar.erase(bar.begin(), lIt);
                          };
    auto shrinkRightFunc = [](size_t cntLim, std::map<float, size_t>& bar)
                           {
                             size_t zCnt = 0;
                             auto rIt = bar.rbegin();
                             while (rIt != bar.rend() && rIt->second < cntLim)
                             {
                               ++zCnt;
                               ++rIt;
                             }
                             auto rIt2 = bar.begin();
                             std::advance(rIt2, bar.size()-zCnt);
                             bar.erase(rIt2, bar.end());
                           };
    shrinkLeftFunc(1, bar);
    shrinkRightFunc(1, bar);
    std::cout << "  " << target_dimension << " dimension bar-chart" << std::endl;
    std::cout << std::setprecision(3);
    for (auto it = bar.begin(), itEnd = bar.end(); it != itEnd; ++it)
      std::cout << "    " << std::setw(5) << it->first << "\t" << it->second << std::endl;
    std::cout << std::setprecision(6);
    // TODO: посчитать попарную корреляцию измерений (измерения-дубликаты)
    // если два измерения почти одинаково упорядочивают слова, то одно из них избыточно (можно сжать модель или доучить измерение)
  } // method-end

  void test_russe2015() const
  {
    std::cout << "RUSSE 2015 evaluation" << std::endl;
    //test_russe2015_dbg();
    test_russe2015_hj();
    test_russe2015_rt();
    test_russe2015_ae();
    test_russe2015_ae2();
  }

  struct SimUsimPredict
  {
    float sim;
    float usim;
    size_t predict;
  };

  std::map<std::string, std::map<std::string, SimUsimPredict>> read_test_file(const std::string& test_file_name, bool with_data = true) const
  {
    size_t correct_fields_cnt = (with_data ? 3 : 2);
    std::map<std::string, std::map<std::string, SimUsimPredict>> test_data;
    std::ifstream ifs(test_file_name.c_str());
    if (!ifs.good()) return test_data;
    std::string line;
    std::getline(ifs, line); // read header
    while ( std::getline(ifs, line).good() )
    {
      if (russe_replace_yo)
        line = std::regex_replace(line, std::regex("ё"), "е");
      const std::regex space_re(",");
      std::vector<std::string> record {
          std::sregex_token_iterator(line.cbegin(), line.cend(), space_re, -1),
          std::sregex_token_iterator()
      };
      if (record.size() != correct_fields_cnt) // invalid record
        continue;
      if (with_data)
        test_data[ record[0] ][ record[1] ].sim = std::stof(record[2]);
      else
        test_data[ record[0] ][ record[1] ];
    }
    return test_data;
  }

  std::optional<float> calc_sim_strong(SimilarityEstimator::CmpDims dims, const std::string& w1, const std::string& w2) const
  {
    return sim_meter->get_sim(dims, w1, w2);
  }

  std::optional<float> calc_sim_with_pn(SimilarityEstimator::CmpDims dims, const std::string& w1, const std::string& w2) const
  {
    const float MINSIM = -1000000;
    std::optional<float> best;
    auto sim1 = sim_meter->get_sim(dims, w1, w2);
    if ( sim1 && sim1.value() > best.value_or(MINSIM))
      best = sim1;
    auto sim2 = sim_meter->get_sim(dims, w1+"_PN", w2);
    if ( sim2 && sim2.value() > best.value_or(MINSIM))
      best = sim2;
    auto sim3 = sim_meter->get_sim(dims, w1, w2+"_PN");
    if ( sim3 && sim3.value() > best.value_or(MINSIM))
      best = sim3;
    auto sim4 = sim_meter->get_sim(dims, w1+"_PN", w2+"_PN");
    if ( sim4 && sim4.value() > best.value_or(MINSIM))
      best = sim4;
    return best;
  }

  void calc_usim(SimilarityEstimator::CmpDims dims, std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    size_t not_found = 0, found = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        auto sim = calc_sim_strong(dims, p1.first, p2.first);
        //auto sim = calc_sim_with_pn(dims, p1.first, p2.first);
        if (sim)
        {
          ++found;
          p2.second.usim = sim.value();
        }
        else
        {
          ++not_found;
          p2.second.usim = 0.0;
          #ifdef DBGIT_NF
            log(p1.first + "," + p2.first);
          #endif
        }
      }
    std::cout << "    not found: " << not_found << " of " << (found+not_found) << " (~" << (not_found*100/(found+not_found)) << "%),      used: " << found << std::endl;
  }

  void calc_predict(std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    for (auto& p1 : test_data)
    {
      std::multimap<float, std::string, std::greater<float>> sorter;
      for (auto& p2 : p1.second)
        sorter.insert( std::make_pair(p2.second.usim, p2.first) );
      size_t half = sorter.size() / 2;
      size_t i = 0;
      for (auto& s : sorter)
      {
        p1.second[s.second].predict = ( i<half ? 1 : 0 );
        ++i;
      }
    }
  }

  float average_precision_light(const std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    std::multiset< std::pair<float, float>, std::greater<std::pair<float, float>>> sorter;
    size_t positive_true = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        sorter.insert( std::make_pair(p2.second.usim, p2.second.sim) );
        if (p2.second.sim == 1)
          ++positive_true;
      }
    float precision_sum = 0;
    float positive_cnt = 0;
    size_t cnt = 0;
    for (auto& s : sorter)
    {
      ++cnt;
      if (s.second == 1)
      {
        positive_cnt += 1;
        precision_sum += positive_cnt / cnt;
      }
    }
    return precision_sum / positive_true;
  }

  float average_precision_sklearn_bin(const std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    std::multiset< std::pair<float, float>, std::greater<std::pair<float, float>>> sorter;
    size_t positive_true = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        sorter.insert( std::make_pair(p2.second.usim, p2.second.sim) );
        if (p2.second.sim == 1)
          ++positive_true;
      }
    float average_precision = 0;
    float last_recall = 0;
    float positive_cnt = 0;
    size_t cnt = 0;
    for (auto& s : sorter)
    {
      ++cnt;
      if (s.second == 1)
      {
        positive_cnt += 1;
        float current_recall = positive_cnt / positive_true;
        float recall_delta = current_recall - last_recall;
        float current_precision = positive_cnt / cnt;
        average_precision += recall_delta * current_precision;
        last_recall = current_recall;
        if (positive_cnt == positive_true)
          break;
      }
    }
    return average_precision;
  }

  float average_precision_sklearn_0_18_bin(const std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    // версия вычисления Average Precision на базе метода трапеций (фактически AUC)
    // использовалась в scikit-learn до версии 0.19.X и в частности в рамках RUSSE-2015
    // современная реализация функции average_precision_score в scikit-learn является прямоугольной аппроксимацией AUC
    // о различиях см. https://datascience.stackexchange.com/questions/52130/about-sklearn-metrics-average-precision-score-documentation
    std::multiset< std::pair<float, float>, std::greater<std::pair<float, float>>> sorter;
    size_t positive_true = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        sorter.insert( std::make_pair(p2.second.usim, p2.second.sim) );
        if (p2.second.sim == 1)
          ++positive_true;
      }
    float average_precision = 0;
    float last_recall = 0;
    float last_precision = 1;
    float positive_cnt = 0;
    size_t cnt = 0;
    for (auto& s : sorter)
    {
      ++cnt;
      if (s.second == 1)
        positive_cnt += 1;
      float current_recall = positive_cnt / positive_true;
      float recall_delta = current_recall - last_recall;
      float current_precision = positive_cnt / cnt;
      average_precision += recall_delta * (last_precision + current_precision)/2;
      last_recall = current_recall;
      last_precision = current_precision;
    }
    return average_precision;
  }

  float accuracy(const std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    size_t succ = 0, total = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        ++total;
        if (p2.second.sim == p2.second.predict)
          ++succ;
      }
    return (float)succ / (float)total;
  }

  float pearsons_rank_correlation_coefficient(const std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    // вычислим мат.ожидания для каждого ряда данных
    float avg_sim = 0, avg_usim = 0, total = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        total += 1;
        avg_sim += p2.second.sim;
        avg_usim += p2.second.usim;
      }
    avg_sim /= total;
    avg_usim /= total;
    // вычислим ковариацию
    float covariance = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
        covariance += (p2.second.sim - avg_sim) * (p2.second.usim - avg_usim);
    covariance /= total;
    // вычислим стандартные отклонения
    float sd1 = 0, sd2 = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        sd1 += (p2.second.sim - avg_sim) * (p2.second.sim - avg_sim);
        sd2 += (p2.second.usim - avg_usim) * (p2.second.usim - avg_usim);
      }
    sd1 = std::sqrt(sd1/total);       // оценка стандартного отклонения на основании смещённой оценки дисперсии, см. https://ru.wikipedia.org/wiki/Среднеквадратическое_отклонение
    sd2 = std::sqrt(sd2/total);
    return covariance / (sd1 * sd2);
  }

  float spearmans_rank_correlation_coefficient(const std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    std::map<std::string, std::map<std::string, SimUsimPredict>> test_data_ranks;

    std::multimap<float, std::pair<std::string, std::string>, std::greater<float>> sorter1, sorter2;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        sorter1.insert( std::make_pair(p2.second.sim, std::make_pair(p1.first, p2.first)) );
        sorter2.insert( std::make_pair(p2.second.usim, std::make_pair(p1.first, p2.first)) );
      }
    // вычисляем дробные ранги (см. https://en.wikipedia.org/wiki/Ranking#Fractional_ranking_.28.221_2.5_2.5_4.22_ranking.29)
    {
      int rank = 0;
      auto sIt = sorter1.begin();
      while ( sIt != sorter1.end() )
      {
        ++rank;
        auto range = sorter1.equal_range(sIt->first);
        size_t cnt = std::distance(range.first, range.second);
        int rank_sum = 0;
        for (size_t idx = 0; idx < cnt; ++idx)
          rank_sum += (rank+idx);
        float fractional_rank = (float)rank_sum / cnt;
        while (range.first != range.second)
        {
          test_data_ranks[range.first->second.first][range.first->second.second].sim = fractional_rank;
          ++range.first;
        }
        rank += (cnt-1);
        sIt = range.second;
      }
    }
    {
      int rank = 0;
      auto sIt = sorter2.begin();
      while ( sIt != sorter2.end() )
      {
        ++rank;
        auto range = sorter2.equal_range(sIt->first);
        size_t cnt = std::distance(range.first, range.second);
        int rank_sum = 0;
        for (size_t idx = 0; idx < cnt; ++idx)
          rank_sum += (rank+idx);
        float fractional_rank = (float)rank_sum / cnt;
        while (range.first != range.second)
        {
          test_data_ranks[range.first->second.first][range.first->second.second].usim = fractional_rank;
          ++range.first;
        }
        rank += (cnt-1);
        sIt = range.second;
      }
    }
    // вычисление коэф.ранговой кореляции Спирмена (см. https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
    return pearsons_rank_correlation_coefficient(test_data_ranks);
  }

  void test_russe2015_dbg() const
  {
    auto test_data = read_test_file("russe2015data/test.csv", false);
    calc_usim(SimilarityEstimator::cdAll, test_data);
    std::ofstream ofs("russe2015data/test_it.csv");
    ofs << "word1,word2,sim" << std::endl;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
        ofs << p1.first << "," << p2.first << "," << p2.second.usim << std::endl;
  }

  void test_russe2015_hj() const
  {
    std::cout << "  HJ" << std::endl;
    auto test_data = read_test_file("russe2015data/hj-test.csv");
    calc_usim(SimilarityEstimator::cdAll, test_data);
    std::cout << "    Use all vector:" << std::endl;
    std::cout << "      Spearman's correlation with human judgements: = " << spearmans_rank_correlation_coefficient(test_data) << std::endl;
    //std::cout << "      Pearson's correlation with human judgements: = " << pearsons_rank_correlation_coefficient(test_data) << std::endl;
    calc_usim(SimilarityEstimator::cdDepOnly, test_data);
    std::cout << "    Use dependency part of vector only:" << std::endl;
    std::cout << "      Spearman's correlation with human judgements: = " << spearmans_rank_correlation_coefficient(test_data) << std::endl;
    calc_usim(SimilarityEstimator::cdAssocOnly, test_data);
    std::cout << "    Use associative part of vector only:" << std::endl;
    std::cout << "      Spearman's correlation with human judgements: = " << spearmans_rank_correlation_coefficient(test_data) << std::endl;
  }

  void test_russe2015_rt() const
  {
    std::cout << "  RT" << std::endl;
    auto test_data = read_test_file("russe2015data/rt-test.csv");
    calc_usim(SimilarityEstimator::cdDepOnly, test_data);
    calc_predict(test_data);
    std::cout << "    Use dependency part of vector only:" << std::endl;
    std::cout << "      average_precision_2015 = " << average_precision_sklearn_0_18_bin(test_data) << "   (used np.trapz)" << std::endl;
    std::cout << "      accuracy = " << accuracy(test_data) << std::endl;
    std::cout << "      average_precision = " << average_precision_sklearn_bin(test_data) << std::endl;
  }


  void test_russe2015_ae() const
  {
    std::cout << "  AE" << std::endl;
    auto test_data = read_test_file("russe2015data/ae-test.csv");
    std::cout << "    Use associative part of vector only:" << std::endl;
    calc_usim(SimilarityEstimator::cdAssocOnly, test_data);
    calc_predict(test_data);
    std::cout << "      average_precision_2015 = " << average_precision_sklearn_0_18_bin(test_data) << "   (used np.trapz)" << std::endl;
    std::cout << "      accuracy = " << accuracy(test_data) << std::endl;
    std::cout << "      average_precision = " << average_precision_sklearn_bin(test_data) << std::endl;
    std::cout << "    Use all vector:" << std::endl;
    calc_usim(SimilarityEstimator::cdAll, test_data);
    calc_predict(test_data);
    std::cout << "      average_precision_2015 = " << average_precision_sklearn_0_18_bin(test_data) << "   (used np.trapz)" << std::endl;
    std::cout << "      accuracy = " << accuracy(test_data) << std::endl;
    std::cout << "      average_precision = " << average_precision_sklearn_bin(test_data) << std::endl;
  }


  void test_russe2015_ae2() const
  {
    std::cout << "  AE2" << std::endl;
    auto test_data = read_test_file("russe2015data/ae2-test.csv");
    std::cout << "    Use associative part of vector only:" << std::endl;
    calc_usim(SimilarityEstimator::cdAssocOnly, test_data);
    calc_predict(test_data);
    std::cout << "      average_precision_2015 = " << average_precision_sklearn_0_18_bin(test_data) << "   (used np.trapz)" << std::endl;
    std::cout << "      accuracy = " << accuracy(test_data) << std::endl;
    std::cout << "      average_precision = " << average_precision_sklearn_bin(test_data) << std::endl;
    std::cout << "    Use all vector:" << std::endl;
    calc_usim(SimilarityEstimator::cdAll, test_data);
    calc_predict(test_data);
    std::cout << "      average_precision_2015 = " << average_precision_sklearn_0_18_bin(test_data) << "   (used np.trapz)" << std::endl;
    std::cout << "      accuracy = " << accuracy(test_data) << std::endl;
    std::cout << "      average_precision = " << average_precision_sklearn_bin(test_data) << std::endl;
  }

  void test_rusim() const
  {
    std::cout << "rusim1000 dataset evaluation" << std::endl;
    {
      std::cout << "  RuSim1000" << std::endl;
      auto test_data = read_test_file("rusim1000data/RuSim1000.csv");
      calc_usim(SimilarityEstimator::cdDepOnly, test_data);
      calc_predict(test_data);
      std::cout << "    Use dependency part of vector only:" << std::endl;
      std::cout << "      average_precision_2015 = " << average_precision_sklearn_0_18_bin(test_data) << "   (used np.trapz)" << std::endl;
      std::cout << "      accuracy = " << accuracy(test_data) << std::endl;
      std::cout << "      average_precision = " << average_precision_sklearn_bin(test_data) << std::endl;
    }
    {
      std::cout << "  RuSim1000-876" << std::endl;
      auto test_data = read_test_file("rusim1000data/RuSim1000-876.csv");
      calc_usim(SimilarityEstimator::cdDepOnly, test_data);
      calc_predict(test_data);
      std::cout << "    Use dependency part of vector only:" << std::endl;
      std::cout << "      average_precision_2015 = " << average_precision_sklearn_0_18_bin(test_data) << "   (used np.trapz)" << std::endl;
      std::cout << "      accuracy = " << accuracy(test_data) << std::endl;
      std::cout << "      average_precision = " << average_precision_sklearn_bin(test_data) << std::endl;
    }
  }

}; // class-decl-end


#endif /* SELFTEST_RU_H_ */
