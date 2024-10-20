#ifndef ORIGINAL_WORD2VEC_VOCABULARY_H_
#define ORIGINAL_WORD2VEC_VOCABULARY_H_

#include "vocabulary.h"
#include "vectors_model.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <regex>
#include <limits>

class OriginalWord2VecVocabulary : public CustomVocabulary
{
public:
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
public:
  // конструктор
  OriginalWord2VecVocabulary()
  : CustomVocabulary()
  {
    vocabulary_hash.reserve(21000000);  // изначально устанавливаем размер хэш-отображения таковым, чтобы эффективно хранить 21 млн. элементов
  }

  // деструктор
  virtual ~OriginalWord2VecVocabulary()
  {
  }

  // функция загрузки словаря из файла
  // предполагается, что словарь отсортирован по убыванию частоты встречаемости слов
  bool load(const std::string& filename)
  {
    // считываем словарь из файла
    std::ifstream ifs( filename );
    if (!ifs.good())
    {
      std::cerr << "Can't open vocabulary file: " << filename << std::endl;
      return false;
    }
    std::string buf;
    while ( std::getline(ifs, buf).good() )
    {
      // каждая запись словаря содержит слово (строку) и абсолютную частоту встречаемости данного слова в корпусе (на основе которого построен словарь)
      // элементы словарной записи разделены пробелами
      const std::regex space_re("\\s+");
      std::vector<std::string> vocabulary_record_components {
          std::sregex_token_iterator(buf.cbegin(), buf.cend(), space_re, -1),
          std::sregex_token_iterator()
      };
      if (vocabulary_record_components.size() != 2)
      {
        std::cerr << "Vocabulary loading error: " << filename << std::endl;
        std::cerr << "Invalid record: " << buf << std::endl;
        return false;
      }
      if ( !whitelist.empty() && whitelist.find(vocabulary_record_components[0]) == whitelist.end() )
        continue;
      if ( stoplist.find(vocabulary_record_components[0]) != stoplist.end() )
        continue;
      vocabulary_hash[vocabulary_record_components[0]] = vocabulary.size(); // сразу строим хэш-отображение для поиска индекса слова в словаре по слову (строке)
      vocabulary.emplace_back( vocabulary_record_components[0], std::stoull(vocabulary_record_components[1]) );
    }
    return true;
  }

  // получение индекса в словаре по тексту слова
  size_t word_to_idx(const std::string& word) const
  {
    auto it = vocabulary_hash.find(word);
    if (it == vocabulary_hash.end())
      return INVALID_IDX;
    else
      return it->second;
  }

  // добавление записи в словарь
  void append(const std::string& word, uint64_t cn)
  {
    vocabulary_hash[word] = vocabulary.size();
    CustomVocabulary::append(word, cn);
  }

  // инициализация списка стоп-слов
  void init_stoplist(const std::string& stopwords_filename)
  {
    std::ifstream ifs(stopwords_filename);
    std::string line;
    while ( std::getline(ifs, line).good() )
      stoplist.insert(line);

  } // method-end

  // инициализация белого списка
  void init_whitelist(const VectorsModel& vm)
  {
    std::copy(vm.vocab.begin(), vm.vocab.end(), std::inserter(whitelist, whitelist.begin()));
  }

  void reset_whitelist()
  {
    whitelist.clear();
  }

  // функция проверки соответствия словаря и модели
  bool check_aligned(const VectorsModel& vm)
  {
    if ( vm.words_count != vocabulary.size() )
    {
      std::cerr << "Model vs. Vocabulary misalignment (size)" << std::endl;
      return false;
    }
    for (size_t i = 0; i < vm.words_count; ++i)
    {
      if ( word_to_idx( vm.vocab[i] ) == INVALID_IDX )
      {
        std::cerr << "Model vs. Vocabulary misalignment: " << vm.vocab[i] << "   " << vocabulary[i].word << std::endl;
        return false;
      }
    }
    return true;
  }

private:
  // хэш-отображение слов в их индексы в словаре (для быстрого поиска)
  std::unordered_map<std::string, size_t> vocabulary_hash;
  // список стоп-слов для словаря (используется при загрузке)
  std::set<std::string> stoplist;
  // белый список для словаря (используется при загрузке)
  std::set<std::string> whitelist;
};

#endif /* ORIGINAL_WORD2VEC_VOCABULARY_H_ */
