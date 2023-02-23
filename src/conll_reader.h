#ifndef CONLL_READER_H_
#define CONLL_READER_H_

#include "str_conv.h"

#include <string>
#include <vector>
#include <algorithm>
#include <fstream>


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
  // конструктор
  ConllReader(const std::string& fn)
  : filename(fn)
  {
    buf = new char[BUF_SIZE];
  }
  ~ConllReader()
  {
    delete[] buf;
  }
  // инициализация
  bool init()
  {
    idx_in_buf = BUF_SIZE;
    real_buf_len = BUF_SIZE;
    if ( filename == "stdin" )
      f = stdin;
    else
      f = fopen(filename.c_str(), "rb");
    if ( f == nullptr )
    {
      std::cerr << "ConllReader error: " << std::strerror(errno) << std::endl;
      return false;
    }
    return true;
  }
  // инициализация (версия для многопоточного чтения)
  bool init_multithread(size_t thread_no, size_t threads_count)
  {
    idx_in_buf = BUF_SIZE;
    real_buf_len = BUF_SIZE;
    uint64_t f_size = 0;
    try {
      f_size = get_file_size();
    } catch (const std::runtime_error& e) {
      std::cerr << "ConllReader can't get file size for: " << filename << "\n  " << e.what() << std::endl;
      return false;
    }
    if (f_size == 0)
    {
      std::cerr << "ConllReader: empty file" << std::endl;
      return false;
    }
    f = fopen(filename.c_str(), "rb");
    if ( f == nullptr )
    {
      std::cerr << "ConllReader error: " << std::strerror(errno) << std::endl;
      return false;
    }
    if ( thread_no != 0 )
    {
      int succ = fseek(f, f_size / threads_count * thread_no, SEEK_SET);
      if (succ != 0)
      {
        std::cerr << "ConllReader error: " << std::strerror(errno) << std::endl;
        return false;
      }
      // т.к. после смещения мы типично не оказываемся в начале предложения, выполним выравнивание на начало предложения
      SentenceMatrix stub;
      read_sentence(stub); // один read_sentence не гарантирует выход на начало предложения, т.к. fseek может поставить нас прямо на перевод строки в конце очередного токена, что распознается, как пустая строка
      stub.clear();
      read_sentence(stub);
    }
    return true;
  }
  // финализация
  void fin()
  {
    fclose( f );
    f = nullptr;
  }
  // чтение предложения
  bool read_sentence(SentenceMatrix& result)
  {
    result.clear();
    if ( !f ) return false;
    while ( true )
    {
      if ( idx_in_buf == real_buf_len && feof(f) ) return false; // больше нечего читать
      if ( ferror(f) ) return false; // больше нет возможности читать
      if ( !read_sentence_internal(result) ) continue; // невалидные предложения пропускаем
      if ( result.empty() ) continue; // пустые предложения пропускаем
      // проконтролируем, что номер первого токена равен единице
      try {
        int tn = std::stoi( result[0][Conll::ID] );
        if (tn != 1) continue;
      } catch (...) {
        continue;
      }
      // проконтролируем нумерацию токенов
      if ( !is_token_no_sequence_valid(result) )
        continue;
      return true;
    }
  }
  // чтение предложения с конвертацией строк к u32string
  bool read_sentence_u32(u32SentenceMatrix& result)
  {
    result.clear();
    SentenceMatrix sentence_matrix;
    if ( !read_sentence(sentence_matrix) ) return false;
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

private:
  // имя conll-файла для чтения
  std::string filename;
  // файловый дескриптор
  FILE* f = nullptr;
  // буфер для чтения
  static constexpr size_t BUF_SIZE = 10 * 1024 * 1024;
  char* buf = nullptr;
  // текущая позиция в буфере для чтения
  size_t idx_in_buf = BUF_SIZE;
  // значимое количество символов в буфере для чтения
  size_t real_buf_len = BUF_SIZE;

  // получение размера файла
  uint64_t get_file_size()
  {
    // TODO: в будущем использовать std::experimental::filesystem::file_size
    std::ifstream ifs(filename, std::ios::binary|std::ios::ate);
    if ( !ifs.good() )
        throw std::runtime_error(std::strerror(errno));
    return ifs.tellg();
  } // method-end
  // чтение строки
  void read_line(FILE *f, std::string& result)
  {
    result.clear();
    while (true)
    {
      if ( idx_in_buf == real_buf_len )
      {
        if ( feof(f) || ferror(f) )
          return;
        idx_in_buf = 0;
        real_buf_len = fread( buf, sizeof(buf[0]), BUF_SIZE, f );
      }
      // согласно принципам кодирования https://ru.wikipedia.org/wiki/UTF-8, никакой другой символ не может содержать в себе байт 0x0A
      // поэтому поиск соответствующего байта является безопасным split-алгоритмом
      while ( idx_in_buf < real_buf_len && buf[idx_in_buf] != '\n' )
        result.push_back( buf[idx_in_buf++] );
      if ( idx_in_buf < real_buf_len && buf[idx_in_buf] == '\n' )
      {
        ++idx_in_buf;
        return;
      }
    }
  } // method-end
  // чтение предложения
  bool read_sentence_internal(SentenceMatrix& result)
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
  // проверка корректности нумерации токенов в предложении
  bool is_token_no_sequence_valid(const SentenceMatrix& data)
  {
    size_t pre_tn = 0;
    for (auto& t : data)
    {
      size_t tn = 0;
      try { tn = std::stoi(t[Conll::ID]); } catch (...) { return false; }
      if (tn != pre_tn+1) return false;
      pre_tn = tn;
    }
    return true;
  } // method-end
}; // class-decl-end


#endif /* CONLL_READER_H_ */
