#ifndef SELFTEST_RU_H_
#define SELFTEST_RU_H_

#include "sim_estimator.h"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <regex>

// процедура оценки качества модели для русского языка (быстрая самодиагностика)
class SelfTest_ru
{
public:
  SelfTest_ru( std::shared_ptr<SimilarityEstimator> sim_estimator)
  : sim_meter(sim_estimator)
  {
  }
  void run(bool verbose = false)
  {
    test_nonsim_dep(verbose);
    std::cout << std::endl;
    test_nonsim_assoc(verbose);
    std::cout << std::endl;
    test_sim_dep(verbose);
    std::cout << std::endl;
    test_sim_assoc(verbose);
//    std::cout << std::endl;
//    test_sim_all(verbose);
//    std::cout << std::endl;
//    dimensions_analyse(verbose);
    std::cout << std::endl;
    test_russe2015();
    std::cout << std::endl;
    test_rusim();
  } // method-end
private:
  // указатель на объект для оценки семантической близости
  std::shared_ptr<SimilarityEstimator> sim_meter;

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
        {"он", "бежать"},
        {"он", "президент"},
        {"они", "быстро"},
        {"они", "купить"},
        {"они", "мочь"},
        {"они", "американец"},
        {"математика", "быстро"},
        {"математика", "во-первых"},
        {"математика", "кофе"},
        {"бежать", "кофе"},
        {"бежать", "во-первых"},
        {"бежать", "вторник"},
        {"бежать", "только"},
        {"бежать", "автомобиль"},
        {"молоко", "митинг"},
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
        {"записка", "служить"},
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

  void calc_usim(SimilarityEstimator::CmpDims dims, std::map<std::string, std::map<std::string, SimUsimPredict>>& test_data) const
  {
    size_t not_found = 0, found = 0;
    for (auto& p1 : test_data)
      for (auto& p2 : p1.second)
      {
        auto sim = sim_meter->get_sim(dims, p1.first, p2.first);
        if (sim)
        {
          ++found;
          p2.second.usim = sim.value();
        }
        else
        {
          ++not_found;
          p2.second.usim = 0.0;
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
