#ifndef FIT_PARUS_H_
#define FIT_PARUS_H_

#include "conll_reader.h"
#include "str_conv.h"

#include <string>
#include <cstring>       // for std::strerror
#include <vector>
#include <set>
#include <fstream>
#include <limits>

class FitParus
{
private:
  typedef std::vector< std::vector<std::string> > SentenceMatrix;
  typedef std::vector< std::vector<std::u32string> > u32SentenceMatrix;
public:
  FitParus()
  : target_column(10)
  {
  }
  // функция запуска преобразования conll-файла
  void run(const std::string& input_fn, const std::string& output_fn)
  {
    // открываем файл с тренировочными данными
    FILE *conll_file = nullptr;
    if ( input_fn == "stdin" )
      conll_file = stdin;
    else
      conll_file = fopen(input_fn.c_str(), "r");
    if ( conll_file == nullptr )
    {
      std::cerr << "Train-file open: error: " << std::strerror(errno) << std::endl;
      return;
    }
    // открываем файл для сохранения результатов
    std::ofstream ofs( output_fn.c_str(), std::ios::binary );   // открываем в бинарном режиме, чтобы в windows не было ретрансляции \n
    if ( !ofs.good() )
    {
      std::cerr << "Resulting-file open: error" << std::endl;
      return;
    }
    // в цикле читаем предложения из CoNLL-файла, преобразуем их и сохраняем в результирующий файл
    SentenceMatrix sentence_matrix;
    u32SentenceMatrix u32_sentence_matrix;
    while ( !feof(conll_file) )
    {
      bool succ = ConllReader::read_sentence(conll_file, sentence_matrix);
      if (!succ)
        continue;
      if (sentence_matrix.size() == 0)
        continue;
//      if ( !is_token_no_sequence_valid(sentence_matrix) )
//        continue;
      // конвертируем строки в utf-32
      u32_sentence_matrix.clear();
      for (auto& t : sentence_matrix)
      {
        u32_sentence_matrix.emplace_back(std::vector<std::u32string>());
        auto& last_token = u32_sentence_matrix.back();
        last_token.reserve(10);
        for (auto& f : t)
          last_token.push_back( StrConv::To_UTF32(f) );
      }
      // выполняем преобразование
      process_sentence(u32_sentence_matrix);
      // конвертируем строки в utf-8
      sentence_matrix.clear();
      for (auto& t : u32_sentence_matrix)
      {
        sentence_matrix.emplace_back(std::vector<std::string>());
        auto& last_token = sentence_matrix.back();
        last_token.reserve(10);
        for (auto& f : t)
          last_token.push_back( StrConv::To_UTF8(f) );
      }
      // сохраняем результат
      save_sentence(ofs, sentence_matrix);
    }
    if ( input_fn != "stdin" )
      fclose(conll_file);
  } // method-end
private:
  // номер conll-колонки, куда записывается результат оптимизации синтаксического контекста
  size_t target_column;
  // сохранение предложения
  void save_sentence(std::ofstream& ofs, const SentenceMatrix& data)
  {
    for (auto& t : data)
    {
      ofs << t[0] << '\t' << t[1] << '\t' << t[2] << '\t' << t[3] << '\t' << t[4] << '\t'
          << t[5] << '\t' << t[6] << '\t' << t[7] << '\t' << t[8] << '\t' << t[9] << '\n';
    }
    ofs << '\n';
  } // method-end
//  // проверка корректности нумерации токенов в предложении
//  bool is_token_no_sequence_valid(const SentenceMatrix& data)
//  {
//    size_t pre_tn = 0;
//    for (auto& t : data)
//    {
//      size_t tn = 0;
//      try { tn = std::stoi(t[0]); } catch (...) { return false; }
//      if (tn != pre_tn+1) return false;
//      pre_tn = tn;
//    }
//    return true;
//  }
  // функция обработки отдельного предложения
  void process_sentence(u32SentenceMatrix& data)
  {
    // приведение токенов к нижнему регистру
    tokens_to_lower(data);
    // эвристика, исправляющая ошибки типизации синатксических связей у знаков препинания
    process_punc(data);
    // неизвестные леммы замещаем на символ подчеркивания (они игнорируются при построении словарей)
    process_unknonw(data);
    // обобщение токенов, содержащих числовые величины
    process_nums(data);
    // фильтрация синтаксических отношений, не заслуживающих внимания
    reltypes_filter(data);
    // поглощение предлогов
    process_prepositions_adv(data);
    process_prepositions(data);
    // преобразование комплетивов
    completive_fit(data);
    // перешагивание через глагол-связку (конструкции с присвязочным отношением)
    process_linking(data);
    // обработка аналитических конструкций (перешагивание через глагол-связку)
    process_analitic(data);
    // обработка конструкций с пассивным залогом
    process_passive(data);
    // развертка сочинительных конструкций
    process_coordinating(data);
    // теоретически, манипуляции со связями (например, с предлогами) могут затереть метку PUNC у списочных знаков препинания
    // запустим принудительную расстановку отношения PUNC повторно
    process_punc(data);
  } // method-end
  // приведение токенов к нижнему регистру
  void tokens_to_lower(u32SentenceMatrix& data)
  {
    for (auto& t : data)
      t[1] = StrConv::toLower(t[1]);
  } // method-end
  // исправление типа синтаксической связи у знаков пунктуации
  void process_punc(u32SentenceMatrix& data)
  {
    std::set<std::u32string> puncts = { U".", U",", U"!", U"?", U":", U";", U"…", U"...", U"--", U"—", U"–", U"‒",
                                        U"'", U"ʼ", U"ˮ", U"\"", U"«", U"»", U"“", U"”", U"„", U"‟", U"‘", U"’", U"‚", U"‛",
                                        U"(", U")", U"[", U"]", U"{", U"}", U"⟨", U"⟩" };
    for (auto& t : data)
    {
      if ( puncts.find(t[1]) != puncts.end() )
        t[7] = U"PUNC";
    }
  } // method-end
  // неизвестные леммы замещаем на символ подчеркивания (они игнорируются при построении словарей)
  void process_unknonw(u32SentenceMatrix& data)
  {
    for (auto& t : data)
      if ( t[2] == U"<unknown>" || t[2] == U"@card@" )
        t[2] = U"_";
  }
  // обобщение токенов, содержащих числовые величины
  void process_nums(u32SentenceMatrix& data)
  {
    // превращаем числа в @num@
//    const std::u32string CARD = U"@card@";
    const std::u32string NUM  = U"@num@";
    const std::u32string Digs = U"0123456789";
    const std::u32string RuLets = U"АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя";
    for (auto& t : data)
    {
      auto& token = t[1];
      auto& lemma = t[2];
      auto& synrel = t[7];
      if (synrel == U"PUNC") continue;
//      // если лемма=@card@ или токен состоит только из цифр, то лемму заменяем на @num@
//      if ( lemma == CARD || token.find_first_not_of(Digs) == std::u32string::npos )
//      {
//        lemma = NUM;
//        continue;
//      }
//      // превращаем 10:10 в @num@:@num@
//      size_t colonPos = token.find(U":");
//      if (colonPos != std::u32string::npos)
//      {
//        std::u32string firstPart  = token.substr(0, colonPos);
//        std::u32string secondPart = token.substr(colonPos+1);
//        if ( firstPart.find_first_not_of(Digs) == std::u32string::npos )
//          if ( secondPart.find_first_not_of(Digs) == std::u32string::npos )
//          {
//            lemma = NUM+U":"+NUM;
//            continue;
//          }
//      }
      // превращаем слова вида 15-летие в @num@-летие
      size_t hyphenPos = token.find(U"-");
      if (hyphenPos != std::u32string::npos)
      {
        std::u32string firstPart = token.substr(0, hyphenPos);
        std::u32string secondPart = token.substr(hyphenPos+1);
        if ( firstPart.find_first_not_of(Digs) == std::u32string::npos )
          if ( secondPart.find_first_not_of(RuLets) == std::u32string::npos )
          {
            size_t lemmaHp = lemma.find(U"-");
            if (lemmaHp != std::u32string::npos)
              lemma = NUM + lemma.substr(lemmaHp);
          }
      }
    } // for all tokens in sentence
  } // method-end
  void reltypes_filter(u32SentenceMatrix& data)
  {
    const std::set<std::u32string> permissible_reltypes = {
        U"предик", U"агент", U"квазиагент", U"дат-субъект",
        U"присвяз", U"аналит", U"пасс-анал",
        U"1-компл", U"2-компл", U"3-компл", U"4-компл", U"неакт-компл",
        // U"сочин", U"соч-союзн", U"кратн",
        U"сочин", U"соч-союзн",
        U"предл",
        U"атриб", U"опред", U"оп-опред",
        U"обст", U"обст-тавт", U"суб-обст", U"об-обст", U"длительн", U"кратно-длительн", U"дистанц",
        U"аппоз", U"количест",
        U"PUNC"
      };
    for (auto& t : data)
    {
      if ( permissible_reltypes.find(t[7]) == permissible_reltypes.end() )
      {
        t[6] = U"0";
        t[7] = U"_";
      }
    }
  } // method-end
  // поглощение предлогов
  void process_prepositions(u32SentenceMatrix& data)
  {
    for (auto& t : data)
    {
      if ( t[7] == U"предл" )
      {
        size_t prepos_token_no = std::stoi( StrConv::To_UTF8(t[6]) );
        if ( prepos_token_no < 1 || prepos_token_no > data.size() )
          continue;
        auto& prepos_token = data[ prepos_token_no - 1  ];
        if (prepos_token[7] == U"сочин" || prepos_token[7] == U"соч-союзн")
        {
          t[6] = U"0";
          t[7] = U"_";
        }
        else
        {
          t[6] = prepos_token[6];
          t[7] = prepos_token[7];
        }
        // prepos_token[6] =  U"0";
        // prepos_token[7] =  U"_";
        prepos_token[6] =  t[0];
        prepos_token[7] =  U"ud_prepos";
      }
    }
  } // method-end
  // поглощение предлогов (продвинутая версия)
  void process_prepositions_adv(u32SentenceMatrix& data)
  {
    const std::map<std::u32string, std::u32string> COMPRESSOR = {
        {U"вроде__g", U"подобно"},
        {U"навроде__g", U"подобно"},
        {U"наподобие__g", U"подобно"},
        {U"подобно__d", U"подобно"},
        {U"сродни__d", U"подобно"},
        {U"а-ля__n", U"подобно"},
        {U"типа__g", U"подобно"},
        {U"сообразно__d", U"сообразно"},
        {U"соответственно__d", U"соответственно"},
        {U"без__g", U"без"},
        {U"безо__g", U"без"},
        {U"в__a", U"в__acc"},
        {U"во__a", U"в__acc"},
        {U"в__l", U"в__loc"},
        {U"во__l", U"в__loc"},
        {U"около__g", U"около"},
        {U"возле__g", U"около"},
        {U"подле__g", U"около"},
        {U"вблизи__g", U"около"},
        {U"близ__g", U"около"},
        {U"порядка__g", U"около"},
        {U"у__g", U"у"},
        {U"обок__g", U"обок"},
        {U"обочь__g", U"обок"},
        {U"вокруг__g", U"вокруг"},
        {U"вкруг__g", U"вокруг"},
        {U"округ__g", U"вокруг"},
        {U"кругом__g", U"вокруг"},
        {U"окрест__g", U"вокруг"},
        {U"благодаря__d", U"благодаря"},
        {U"ввиду__g", U"вследствие"},
        {U"вследствие__g", U"вследствие"},
        {U"исключая__a", U"исключая"},
        {U"выключая__a", U"исключая"},
        {U"внутри__g", U"внутри"},
        {U"внутрь__g", U"внутрь"},
        {U"вовнутрь__g", U"внутрь"},
        {U"вглубь__g", U"вглубь"},
        {U"вне__g", U"вне"},
        {U"снаружи__g", U"снаружи"},
        {U"накануне__g", U"накануне"},
        {U"вдоль__g", U"вдоль"},
        {U"вместо__g", U"вместо"},
        {U"замест__g", U"вместо"},
        {U"заместо__g", U"вместо"},
        {U"наместо__g", U"вместо"},
        {U"взамен__g", U"взамен"},
        {U"включая__a", U"включая"},
        {U"кончая__i", U"кончая"},
        {U"перед__i", U"перед"},
        {U"передо__i", U"перед"},
        {U"пред__i", U"перед"},
        {U"предо__i", U"перед"},
        {U"вперед__g", U"впереди"},
        {U"впереди__g", U"впереди"},
        {U"поперед__g", U"впереди"},
        {U"позади__g", U"позади"},
        {U"сзади__g", U"позади"},
        {U"назади__g", U"позади"},
        {U"после__g", U"после"},
        {U"опосля__g", U"после"},
        {U"позднее__g", U"после"},
        {U"спустя__a", U"спустя"},
        {U"вслед__d", U"вслед"},
        {U"вослед__d", U"вслед"},
        {U"вдогон__d", U"вдогон"},
        {U"для__g", U"для"},
        {U"ради__g", U"для"},
        {U"до__g", U"до"},
        {U"прежде__g", U"прежде"},
        {U"за__a", U"за__acc"},
        {U"за__i", U"за__ins"},
        {U"из__g", U"из"},
        {U"изо__g", U"из"},
        {U"из-под__g", U"из-под"},
        {U"из-подо__g", U"из-под"},
        {U"изнутри__g", U"изнутри"},
        {U"из-за__g", U"из-за"},
        {U"к__d", U"к"},
        {U"ко__d", U"к"},
        {U"навстречу__d", U"навстречу"},
        {U"мимо__g", U"мимо"},
        {U"на__a", U"на__acc"},
        {U"на__l", U"на__loc"},
        {U"назад__a", U"назад"},
        {U"наперерез__d", U"наперерез"},
        {U"наперехват__d", U"наперерез"},
        {U"о__a", U"о__acc"},
        {U"об__a", U"о__acc"},
        {U"обо__a", U"о__acc"},
        {U"о__l", U"о__loc"},
        {U"об__l", U"о__loc"},
        {U"обо__l", U"о__loc"},
        {U"касаемо__g", U"касаемо"},
        {U"касательно__g", U"касаемо"},
        {U"насчет__g", U"насчет"},
        {U"относительно__g", U"относительно"},
        {U"от__g", U"от"},
        {U"ото__g", U"от"},
        {U"по__a", U"по__acc"},
        {U"по__d", U"по__dat"},
        {U"по__l", U"по__loc"},
        {U"по-за__a", U"по-за__acc"},
        {U"по-за__i", U"по-за__ins"},
        {U"по-под__i", U"по-под"},
        {U"под__a", U"под__acc"},
        {U"подо__a", U"под__acc"},
        {U"под__i", U"под__ins"},
        {U"подо__i", U"под__ins"},
        {U"ниже__g", U"ниже"},
        {U"снизу__g", U"ниже"},
        {U"внизу__g", U"ниже"},
        {U"поперек__g", U"поперек"},
        {U"посредством__g", U"посредством"},
        {U"путем__g", U"посредством"},
        {U"при__l", U"при"},
        {U"про__a", U"про"},
        {U"напротив__g", U"напротив"},
        {U"насупротив__g", U"напротив"},
        {U"против__g", U"против"},
        {U"противу__g", U"против"},
        {U"супротив__g", U"против"},
        {U"вопреки__d", U"вопреки"},
        {U"наперекор__d", U"вопреки"},
        {U"противно__d", U"вопреки"},
        {U"вразрез__d", U"вопреки"},
        {U"с__g", U"с__gen"},
        {U"со__g", U"с__gen"},
        {U"с__a", U"с__acc"},
        {U"со__a", U"с__acc"},
        {U"с__i", U"с__ins"},
        {U"со__i", U"с__ins"},
        {U"кроме__g", U"кроме"},
        {U"окроме__g", U"кроме"},
        {U"окромя__g", U"кроме"},
        {U"опричь__g", U"кроме"},
        {U"помимо__g", U"помимо"},
        {U"сверх__g", U"сверх"},
        {U"свыше__g", U"сверх"},
        {U"над__i", U"над"},
        {U"надо__i", U"над"},
        {U"выше__g", U"выше"},
        {U"превыше__g", U"превыше"},
        {U"вверху__g", U"вверху"},
        {U"наверху__g", U"вверху"},
        {U"сверху__g", U"вверху"},
        {U"поверх__g", U"поверх"},
        {U"посверху__g", U"поверх"},
        {U"сквозь__a", U"сквозь"},
        {U"скрозь__a", U"сквозь"},
        {U"согласно__d", U"согласно"},
        {U"согласно__g", U"согласно"},
        {U"среди__g", U"среди"},
        {U"средь__g", U"среди"},
        {U"середи__g", U"среди"},
        {U"середь__g", U"среди"},
        {U"посередине__g", U"посередине"},
        {U"посередке__g", U"посередине"},
        {U"посередь__g", U"посередине"},
        {U"посреди__g", U"посередине"},
        {U"осредине__g", U"посередине"},
        {U"меж__i", U"между__ins"},
        {U"между__i", U"между__ins"},
        {U"промеж__i", U"между__ins"},
        {U"промежду__i", U"между__ins"},
        {U"меж__g", U"между__gen"},
        {U"между__g", U"между__gen"},
        {U"промеж__g", U"между__gen"},
        {U"промежду__g", U"между__gen"},
        {U"через__a", U"через"},
        {U"черезо__a", U"через"},
        {U"чрез__a", U"через"}
    };
    for (auto& t : data)
    {
      if ( t[7] != U"предл" || t[5].length() == 0 || t[5][0] != U'N' ) continue;
      size_t prepos_token_no = std::stoi( StrConv::To_UTF8(t[6]) );
      if ( prepos_token_no < 1 || prepos_token_no > data.size() ) continue;
      auto& prepos_token = data[ prepos_token_no - 1  ];
      std::u32string case_code = (t[5].length() > 4) ? std::u32string(1, t[5][4]) : U"-";
      std::u32string key = prepos_token[2] + U"__" + case_code;
      auto it = COMPRESSOR.find(key);
      if ( it == COMPRESSOR.end() ) continue;
      if (prepos_token[7] == U"сочин" || prepos_token[7] == U"соч-союзн")
      {
        t[6] = U"0";
        t[7] = U"_";
      }
      else
      {
        t[6] = prepos_token[6];
        t[7] = it->second;
      }
      prepos_token[6] =  t[0];
      prepos_token[7] =  U"ud_prepos";
    }
  } // method-end
  // преобразование комплетивов
  void completive_fit(u32SentenceMatrix& data)
  {
    for (auto& t : data)
    {
      if ( t[5].length() > 0 && t[5][0] == U'N' )
      {
        if ( t[7] == U"2-компл" || t[7] == U"3-компл" || t[7] == U"4-компл" )
        {
          std::u32string::value_type case_code = (t[5].length() > 4) ? t[5][4] : U'-';
          t[7] = U"компл-" + std::u32string(1, case_code);
        }
      }
    }
  }
  // перешагивание через глагол-связку (конструкции с присвязочным отношением к именной части сказуемого или адъективу)
  void process_linking(u32SentenceMatrix& data)
  {
    for (auto& t : data)
    {
      if ( t[7] == U"присвяз" )
      {
        size_t predicate_token_no = find_child(data, t[6], U"предик");
        if ( predicate_token_no == 0 )
          continue;
        auto& predicate_token = data[ predicate_token_no - 1 ];
        if ( t[5].length() > 0 && (t[5][0] == U'N' || t[5][0] == U'A') && predicate_token[5].length() > 0 && predicate_token[5][0] == U'N' )
          predicate_token[6] = t[0];
        // t[6] = U"0";
        // t[7] = U"_";
      }
    }
  } // method-end
  // обработка аналитических конструкций (перешагивание через глагол-связку)
  void process_analitic(u32SentenceMatrix& data)
  {
    for (auto& t : data)
    {
      if ( t[7] == U"аналит" && t[2] != U"бы" && t[2] != U"б" )
      {
        // всех потомков глагола-связки перевесим на содержательный глагол
        for (auto& ti : data)
        {
          if ( ti[6] == t[6] && ti[0] != t[0] && ti[7] != U"присвяз" )
            ti[6] = t[0];
        }
        //t[6] = U"0";
        //t[7] = U"_";
      }
    }
  } // method-end
  // обработка конструкций с пассивным залогом
  void process_passive(u32SentenceMatrix& data)
  {
    for (auto& t : data)
    {
      if ( t[7] == U"пасс-анал" ) // преобразование пассивно-аналитической конструкции
      {
        // всех потомков глагола-связки перевесим на содержательный глагол
        for (auto& ti : data)
        {
          if ( ti[6] == t[6] && ti[0] != t[0] && ti[7] != U"присвяз" )
          {
            ti[6] = t[0];
            if ( ti[7] == U"предик" )
              ti[7] = U"предик-пасс";
          }
        }
        //t[6] = U"0";
        //t[7] = U"_";
      }
      if ( t[7] == U"предик" )
      {
        size_t head_token_no = std::stoi( StrConv::To_UTF8(t[6]) );
        if ( head_token_no < 1 || head_token_no > data.size() )
          continue;
        auto& head_token = data[ head_token_no - 1  ];
        auto& head_msd = head_token[5];
        if ( head_msd.length() >= 8 && head_msd[0] == U'V' && head_msd[2] == U'p' && head_msd[7] == U'p' ) // причастие в пассивном залоге
          t[7] = U"предик-пасс";
      }
    } // for all tokens in sentence
  } // method-end
  // развертывание и чистка сочинительных конструкций
  void process_coordinating(u32SentenceMatrix& data)
  {
    // сначала перебросим соч-союзн мимо союза
    for (auto& t : data)
    {
      if (t[7] != U"соч-союзн") continue;
      if (t[6] == U"0")
      {
        // invalid record
        t[7] = U"_";
        continue;
      }
      size_t head_token_idx = std::stoi( StrConv::To_UTF8(t[6]) ) - 1;
      auto& head_t = data[head_token_idx];
      if (head_t[2] != U"и" || head_t[7] != U"сочин") // другое не интересно
      {
        t[6] = U"0";
        t[7] = U"_";
        continue;
      }
      t[6] = head_t[6];
      t[7] = head_t[7];
    }
    // теперь для сочиненных поставим такую же связь, что и для головы цепочки
    for (auto& t : data)
    {
      if (t[7] != U"сочин") continue;
      bool good_item = t[5].length() > 0 && (t[5][0] == U'N' || t[5][0] == U'A');
      if (!good_item)
      {
        t[6] = U"0";
        t[7] = U"_";
        continue;
      }
      size_t current_token_idx = std::stoi( StrConv::To_UTF8(t[0]) ) - 1;
      size_t chain_head_idx = find_coord_head(data, current_token_idx);
      if (chain_head_idx == std::numeric_limits<size_t>::max())
      {
        t[6] = U"0";
        t[7] = U"_";
        continue;
      }
      auto& head_t = data[chain_head_idx];
      if (head_t[7] == U"аппоз")
      {
        t[6] = U"0";
        t[7] = U"_";
        continue;
      }
      t[6] = head_t[6];
      t[7] = head_t[7];
    }
  }
  // поиск начала сочинительной цепочки
  size_t find_coord_head(const u32SentenceMatrix& data, size_t current_token_idx)
  {
    auto& t = data[current_token_idx];
    if (t[7] != U"сочин")
      return current_token_idx;
    if (t[6] == U"0")
      return std::numeric_limits<size_t>::max(); // invalid record
    size_t head_token_idx = std::stoi( StrConv::To_UTF8(t[6]) ) - 1;
    return find_coord_head(data, head_token_idx);
  }
  // поиск первого потомка с заданным типом отношения к родителю
  size_t find_child(const u32SentenceMatrix& data, const std::u32string& node_no, const std::u32string& rel_type)
  {
    for (auto& t : data)
    {
      if ( t[6] == node_no && t[7] == rel_type )
        return std::stoi( StrConv::To_UTF8(t[0]) );
    }
    return 0;
  } // method-end
}; // class-decl-end


#endif /* FIT_PARUS_H_ */
