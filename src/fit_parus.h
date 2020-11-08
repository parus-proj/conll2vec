#ifndef FIT_PARUS_H_
#define FIT_PARUS_H_

#include "conll_reader.h"
#include "str_conv.h"

#include <string>
#include <cstring>       // for std::strerror
#include <vector>
#include <set>
#include <fstream>

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
    process_prepositions(data);
    // перешагивание через глагол-связку (конструкции с присвязочным отношением)
    process_linking(data);
    // обработка аналитических конструкций (перешагивание через глагол-связку)
    process_analitic(data);
    // обработка конструкций с пассивным залогом
    process_passive(data);
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
      if ( t[2] == U"<unknown>" )
        t[2] = U"_";
  }
  // обобщение токенов, содержащих числовые величины
  void process_nums(u32SentenceMatrix& data)
  {
    // превращаем числа в @num@
    const std::u32string CARD = U"@card@";
    const std::u32string NUM  = U"@num@";
    const std::u32string Digs = U"0123456789";
    const std::u32string RuLets = U"АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя";
    for (auto& t : data)
    {
      auto& token = t[1];
      auto& lemma = t[2];
      auto& synrel = t[7];
      if (synrel == U"PUNC") continue;
      // если лемма=@card@ или токен состоит только из цифр, то лемму заменяем на @num@
      if ( lemma == CARD || token.find_first_not_of(Digs) == std::u32string::npos )
      {
        lemma = NUM;
        continue;
      }
      // превращаем 10:10 в @num@:@num@
      size_t colonPos = token.find(U":");
      if (colonPos != std::u32string::npos)
      {
        std::u32string firstPart  = token.substr(0, colonPos);
        std::u32string secondPart = token.substr(colonPos+1);
        if ( firstPart.find_first_not_of(Digs) == std::u32string::npos )
          if ( secondPart.find_first_not_of(Digs) == std::u32string::npos )
          {
            lemma = NUM+U":"+NUM;
            continue;
          }
      }
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
        t[6] = prepos_token[6];
        t[7] = prepos_token[7];
        // prepos_token[6] =  U"0";
        // prepos_token[7] =  U"_";
        prepos_token[6] =  t[0];
        prepos_token[7] =  U"ud_prepos";
      }
    }
  } // method-end
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
