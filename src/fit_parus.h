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
    FILE *conll_file = fopen(input_fn.c_str(), "rb");
    if ( conll_file == nullptr )
    {
      std::cerr << "Train-file open: error: " << std::strerror(errno) << std::endl;
      return;
    }
    // открываем файл для сохранения результатов
    std::ofstream ofs(output_fn.c_str());
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
    // эвристика, исправляющая ошибки типизации синатксических связей у знаков препинания
    process_punc(data);
    // неизвестные леммы замещаем на символ подчеркивания (они игнорируются при построении словарей)
    process_unknonw(data);
    // обобщение токенов, содержащих числовые величины
    process_nums(data);
  } // method-end
  // исправление типа синтаксической связи у знаков пунктуации
  void process_punc(u32SentenceMatrix& data)
  {
    std::set<std::u32string> puncts = { U".", U",", U"!", U"?", U":", U";", U"…", U"...", U"--", U"'", U"«", U"»", U"(", U")", U"[", U"]", U"{", U"}" };
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
};


#endif /* FIT_PARUS_H_ */
