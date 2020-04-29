#ifndef VOCABS_BUILDER_H_
#define VOCABS_BUILDER_H_

#include "conll_reader.h"

#include <string>
#include <cstring>       // for std::strerror
#include <vector>
#include <unordered_map>

// Класс, обеспечивающие создание словарей (-task vocab)
class VocabsBuilder
{
private:
  typedef std::vector< std::vector<std::string> > SentenceMatrix;
  typedef std::unordered_map<std::string, uint64_t> VocabMapping;
  typedef std::shared_ptr<VocabMapping> VocabMappingPtr;
public:
  // построение всех словарей
  bool build_vocabs(const std::string& conll_fn, const std::string& voc_m_fn, const std::string& voc_p_fn,
                    const std::string& voc_d_fn, const std::string& voc_a_fn,
                    size_t limit_m, size_t limit_p, size_t limit_d, size_t limit_a,
                    size_t embeddings_vocabulary_column, size_t ctx_vocabulary_column_d)
  {
    // открываем файл с тренировочными данными
    FILE *conll_file = fopen(conll_fn.c_str(), "rb");
    if ( conll_file == nullptr )
    {
      std::cerr << "Train-file open: error: " << std::strerror(errno) << std::endl;
      return false;
    }

    // создаем контейенеры для словарей
    VocabMappingPtr vocab_main = std::make_shared<VocabMapping>();
    VocabMappingPtr vocab_proper = std::make_shared<VocabMapping>();
    VocabMappingPtr vocab_dep = std::make_shared<VocabMapping>();
    VocabMappingPtr vocab_assoc = std::make_shared<VocabMapping>();

    // в цикле читаем предложения из CoNLL-файла и извлекаем из них информацию для словарей
    SentenceMatrix sentence_matrix;
    sentence_matrix.reserve(5000);
    uint64_t sr_fails_cnt = 0;    // количество ошибок чтения предложений (предложений, содержащих хотя бы одну некорректную запись)
    uint64_t sentence_processed = 0;
    uint64_t tokens_processed = 0;
    while ( !feof(conll_file) )
    {
      bool succ = ConllReader::read_sentence(conll_file, sentence_matrix);
      tokens_processed += sentence_matrix.size(); // статистику ведём и по некорректным предложениям
      if (sentence_matrix.size() > 0)
        sentence_processed++;
      if (!succ)
      {
        ++sr_fails_cnt;
        continue;
      }
      if (sentence_matrix.size() == 0)
        continue;
      process_sentence_main(vocab_main, sentence_matrix, embeddings_vocabulary_column);
      process_sentence_proper(vocab_proper, sentence_matrix, embeddings_vocabulary_column);
      process_sentence_dep_ctx(vocab_dep, sentence_matrix, ctx_vocabulary_column_d);
      process_sentence_assoc_ctx(vocab_assoc, sentence_matrix, 2); // всегда строим по нормальным формам
    }
    fclose(conll_file);
    if ( sr_fails_cnt > 0)
      std::cerr << "Sentence reading fails count: " << sr_fails_cnt << std::endl;
    std::cout << "Sentences count: " << sentence_processed << std::endl;
    std::cout << "Tokens count: " << tokens_processed << std::endl;
    std::cout << std::endl;

    // сохраняем словари в файлах
    std::cout << "Save main vocabulary..." << std::endl;
    save_vocab(vocab_main, limit_m, voc_m_fn);
    std::cout << "Save proper names vocabulary..." << std::endl;
    save_vocab(vocab_proper, limit_p, voc_p_fn);
    std::cout << "Save dependency contexts vocabulary..." << std::endl;
    save_vocab(vocab_dep, limit_d, voc_d_fn);
    std::cout << "Save associative contexts vocabulary..." << std::endl;
    save_vocab(vocab_assoc, limit_a, voc_a_fn);
    return true;
  } // method-end
private:
  // проверка, является ли токен собственным именем
  bool isProperName(const std::string& feats)
  {
    return feats.length() >=2 && feats[0] == 'N' && feats[1] == 'p';
  } // method-end
  void process_sentence_main(VocabMappingPtr vocab, const SentenceMatrix& sentence, size_t column)
  {
    for (auto& token : sentence)
    {
      if (token[7] == "PUNC")  // знаки препинания в основной словарь не включаем (они обрабатываются особо)
        continue;
      if ( isProperName(token[5]) )
        continue;
      auto&& word = token[column];
      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;
    }
  } // method-end
  void process_sentence_proper(VocabMappingPtr vocab, const SentenceMatrix& sentence, size_t column)
  {
    for (auto& token : sentence)
    {
      if ( !isProperName(token[5]) )
        continue;
      auto&& word = token[column];
      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;
    }
  } // method-end
  void process_sentence_dep_ctx(VocabMappingPtr vocab, const SentenceMatrix& sentence, size_t column)
  {
    for (auto& token : sentence)
    {
      if (token[7] == "PUNC")  // знаки препинания в словарь синтаксических контекстов не включаем
        continue;
      auto&& word = token[column];
      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;
    }
  } // method-end
  void process_sentence_assoc_ctx(VocabMappingPtr vocab, const SentenceMatrix& sentence, size_t column)
  {
    for (auto& token : sentence)
    {
      // TODO: здесь возможна мощная логика фильтрации по стоп-словам (союзы и т.п.)
      if (token[7] == "PUNC")  // знаки препинания в словарь контекстов для моделирования ассоциаций не включаем
        continue;
      auto&& word = token[column];
      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;
    }
  } // method-end
  // редукция и сохранение словаря в файл
  void save_vocab(VocabMappingPtr vocab, size_t min_count, const std::string& file_name)
  {
    // удаляем редкие слова (ниже порога отсечения)
    std::cout << "  min-count reduce" << std::endl;
    auto it = vocab->begin();
    while (it != vocab->end())    //TODO: в C++20 заменить на std::erase_if (https://en.cppreference.com/w/cpp/container/map/erase_if)
    {
      if (it->second >= min_count)
        ++it;
      else
        it = vocab->erase(it);
    }
    std::cout << "  resulting vocabulary size: " << vocab->size() << std::endl;
    // пересортируем в порядке убывания частоты
    std::multimap<uint64_t, std::string, std::greater<uint64_t>> revVocab;
    for (auto& record : *vocab)
      revVocab.insert( std::make_pair(record.second, record.first) );
    // сохраняем словарь в файл
    FILE *fo = fopen(file_name.c_str(), "wb");
    for (auto& record : revVocab)
      fprintf(fo, "%s %lu\n", record.second.c_str(), record.first);
    fclose(fo);
  } // method-end
};


#endif /* VOCABS_BUILDER_H_ */
