#ifndef CONLL_READER_H_
#define CONLL_READER_H_

#include "str_conv.h"

#include <string>
#include <vector>
#include <algorithm>


enum Conll
{
  ID = 0,
  FORM,
  LEMMA,
  UPOS,
  XPOS,
  FEATURES,
  HEAD,
  DEPREL,
  DEPS,
  MISC
};


class ConllReader
{
public:
  typedef std::vector< std::vector<std::string> > SentenceMatrix;
  typedef std::vector< std::vector<std::u32string> > u32SentenceMatrix;
private:
  static constexpr size_t FIELDS_COUNT = 10;
  static constexpr size_t DELIM_COUNT = FIELDS_COUNT - 1;
public:
  // чтение строки
  static void read_line(FILE *f, std::string& result)
  {
    result.clear();
    while (true)
    {
      int c = fgetc(f);
      if ( feof(f) || ferror(f) )
        return;
      // согласно принципам кодирования https://ru.wikipedia.org/wiki/UTF-8, никакой другой символ не может содержать в себе байт 0x0A
      // поэтому поиск соответствующего байта является безопасным split-алгоритмом
      if ( c == '\n' )
        return;
      result.push_back(c);
    }
  } // method-end
  // чтение предложения
  static bool read_sentence(FILE *f, SentenceMatrix& result)
  {
    result.clear();
    bool status = true;
    std::string line;
    line.reserve(1024);
    while (true)
    {
      read_line(f, line);
      if ( !line.empty() && line.back() == '\r' )  // remove 'windows EOL component'
        line.pop_back();
      if ( line.empty() )
        return status;
      if ( line[0] == '#' )  // conll comment
        continue;
      // разбиваем строку по символу табуляции
      // согласно принципам кодирования https://ru.wikipedia.org/wiki/UTF-8, никакой другой символ не может содержать в себе байт 0x09
      // поэтому поиск соответствующего байта является безопасным split-алгоритмом
      size_t delimiters_count = std::count(line.begin(), line.end(), '\t');
      if ( delimiters_count != DELIM_COUNT ) // должно быть 10 полей, т.е. 9 разделителей
        status = false;
      result.emplace_back( std::vector<std::string>(FIELDS_COUNT) );
      auto& last_token = result.back();
      size_t fieldStartPos = 0;
      size_t idx = 0;
      while (true)
      {
        size_t fieldEndPos = line.find('\t', fieldStartPos);
        if ( fieldEndPos != std::string::npos )
        {
          last_token[idx++] = line.substr(fieldStartPos, fieldEndPos-fieldStartPos);
          fieldStartPos = fieldEndPos + 1;
        }
        else
        {
          last_token[idx] = line.substr(fieldStartPos);
          break;
        }
      } // tab split loop
    } // lines read loop
  } // method-end
  // чтение предложения с конвертацией строк к u32string
  static bool read_sentence_u32(FILE *f, u32SentenceMatrix& result)
  {
    result.clear();
    SentenceMatrix sentence_matrix;
    if ( !ConllReader::read_sentence(f, sentence_matrix) ) return false;
    result.reserve( sentence_matrix.size() );
    for (const auto& t : sentence_matrix)
    {
      result.emplace_back( std::vector<std::u32string>(FIELDS_COUNT) );
      auto& last_token = result.back();
      for (size_t i = 0; i < FIELDS_COUNT; ++i)
        last_token[i] = StrConv::To_UTF32(t[i]);
    }
    return true;
  } // method-end
};


#endif /* CONLL_READER_H_ */
