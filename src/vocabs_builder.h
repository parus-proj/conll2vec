#ifndef VOCABS_BUILDER_H_
#define VOCABS_BUILDER_H_

#include "conll_reader.h"
#include "str_conv.h"
#include "categoroid_vocab.h"
#include "mwe_vocabulary.h"
#include "original_word2vec_vocabulary.h"

#include <memory>
#include <string>
#include <cstring>       // for std::strerror
#include <vector>
#include <unordered_map>
#include <set>
#include <fstream>

// Класс, хранящий данные по чтению обучающих данных и выводящий прогресс-сообщения
class StatHelper
{
public:
  void calc_sentence(size_t cnt)
  {
    tokens_processed += cnt; // статистику ведём и по некорректным предложениям
    tokens_dbg_counter += cnt;
    if (tokens_dbg_counter >= 100000)
    {
      tokens_dbg_counter %= 100000;
      if (tokens_processed >= 1000000)
        std::cout << '\r' << (tokens_processed / 1000000) << " M        ";
      else
        std::cout << '\r' << (tokens_processed / 1000) << " K        ";
      std::cout.flush();
    }
    if (cnt > 0)
      sentence_processed++;
  }
  void inc_sr_fils()
  {
    ++sr_fails_cnt;
  }
  void output_stat()
  {
    if ( sr_fails_cnt > 0)
      std::cerr << "Sentence reading fails count: " << sr_fails_cnt << std::endl;
    std::cout << "Sentences count: " << sentence_processed << std::endl;
    std::cout << "Tokens count: " << tokens_processed << std::endl;
    std::cout << std::endl;
  }
private:
  uint64_t sr_fails_cnt = 0;    // количество ошибок чтения предложений (предложений, содержащих хотя бы одну некорректную запись)
  uint64_t sentence_processed = 0;
  uint64_t tokens_processed = 0;
  size_t   tokens_dbg_counter = 0;
};


// Класс, обеспечивающие создание словарей (-task vocab)
class VocabsBuilder
{
private:
  typedef ConllReader::SentenceMatrix SentenceMatrix;
  typedef std::unordered_map<std::string, uint64_t> VocabMapping;
  typedef std::shared_ptr<VocabMapping> VocabMappingPtr;
  typedef std::unordered_map<std::string, std::map<std::string, size_t>> Token2LemmasMap;
  typedef std::shared_ptr<Token2LemmasMap> Token2LemmasMapPtr;
  typedef std::shared_ptr<CategoroidsVocabulary> CategoroidsVocabularyPtr;
public:
  // построение всех словарей
  bool build_vocabs(const std::string& conll_fn,
                    const std::string& voc_l_fn, const std::string& voc_t_fn,
                    const std::string& voc_tm_fn, const std::string& voc_oov_fn, const std::string& voc_d_fn,
                    size_t limit_l, size_t limit_t, size_t limit_o, size_t limit_d,
                    size_t ctx_vocabulary_column_d, bool use_deprel, /*bool excludeNumsFromToks,*/ size_t max_oov_sfx,
                    const std::string& categoroids_vocab_fn, const std::string& mwe_fn)
  {

    // загружаем справочник категороидов (при наличии)
    CategoroidsVocabularyPtr coid_vocab;
    if ( !categoroids_vocab_fn.empty() )
    {
      coid_vocab = std::make_shared<CategoroidsVocabulary>();
      if ( !coid_vocab->load_words_list(categoroids_vocab_fn) )
      {
        std::cerr << "Categoroids-file loading error." << std::endl;
        return false;
      }
    }

    // Проход 1: строим главный словарь (включая словосочетания)

    bool succ = build_main_vocab_only(conll_fn, mwe_fn, voc_l_fn, limit_l, coid_vocab);
    if ( !succ ) return false;

    // Проход2: строим остальные словари уже с учётом того, какие именно словосочетания преодолели частотный порог основного словаря

    // создаём справочник словосочетаний
    std::shared_ptr< OriginalWord2VecVocabulary > v_main = std::make_shared<OriginalWord2VecVocabulary>();
    if ( !v_main->load(voc_l_fn) )
      return false;
    std::shared_ptr< MweVocabulary > v_mwe = std::make_shared<MweVocabulary>();
    if ( !v_mwe->load(mwe_fn, v_main) )
      return false;

    // открываем файл с тренировочными данными
    FILE *conll_file = fopen(conll_fn.c_str(), "rb");
    if ( conll_file == nullptr )
    {
      std::cerr << "Train-file open: error: " << std::strerror(errno) << std::endl;
      return false;
    }

    // создаем контейнеры для словарей
    VocabMappingPtr vocab_token = std::make_shared<VocabMapping>();
    Token2LemmasMapPtr token2lemmas_map = std::make_shared<Token2LemmasMap>();
    VocabMappingPtr vocab_oov = (voc_oov_fn.empty()) ? nullptr : std::make_shared<VocabMapping>();
    VocabMappingPtr vocab_dep = std::make_shared<VocabMapping>();

    // в цикле читаем предложения из CoNLL-файла и извлекаем из них информацию для словарей
    SentenceMatrix sentence_matrix;
    sentence_matrix.reserve(5000);
    StatHelper stat;
    while ( !feof(conll_file) )
    {
      bool succ = ConllReader::read_sentence(conll_file, sentence_matrix);
      stat.calc_sentence(sentence_matrix.size());
      if (!succ)
      {
        stat.inc_sr_fils();
        continue;
      }
      if (sentence_matrix.size() == 0)
        continue;
      apply_patches(sentence_matrix); // todo: УБРАТЬ!  временный дополнительный корректор для борьбы с "грязными данными" в результатах лемматизации
      v_mwe->put_phrases_into_sentence(sentence_matrix);
      process_sentence_tokens(vocab_token, token2lemmas_map, /*excludeNumsFromToks,*/ sentence_matrix);
      if (vocab_oov)
        process_sentence_oov(vocab_oov, sentence_matrix, max_oov_sfx);
      process_sentence_dep_ctx(vocab_dep, sentence_matrix, ctx_vocabulary_column_d, use_deprel);
    }
    fclose(conll_file);
    std::cout << std::endl;
    stat.output_stat();
    // сохраняем словари в файлах
    std::cout << "Save tokens vocabulary..." << std::endl;
    erase_toks_stopwords(vocab_token); // todo:  УБРАТЬ! временный доп.фильтр для борьбы с ошибками токенизации
    reduce_vocab(vocab_token, limit_t);
    save_vocab(vocab_token, voc_t_fn, token2lemmas_map, voc_tm_fn);
    if (vocab_oov)
    {
      std::cout << "Save OOV vocabulary..." << std::endl;
      oov_idf_filter(vocab_oov, vocab_token, max_oov_sfx);
      reduce_vocab(vocab_oov, limit_o);
      save_vocab(vocab_oov, voc_oov_fn);
    }
    std::cout << "Save dependency contexts vocabulary..." << std::endl;
    reduce_vocab(vocab_dep, limit_d);
    save_vocab(vocab_dep, voc_d_fn);
    return true;
  } // method-end
private:
  // маркер oov-слова
  const std::string OOV = "_OOV_";
  // минимальная длина слова, от которого берутся oov-суффиксы
  const size_t SFX_SOURCE_WORD_MIN_LEN = 6;
  // функция построения и сохранения главного словаря
  // выполняется отдельно, т.к. необходимо выяснить частоты словосочетаний (какие из них преодолевают частотный порог главного словаря и будут преобразовываться)
  bool build_main_vocab_only(const std::string& conll_fn, const std::string& mwe_fn, const std::string& voc_l_fn, size_t limit_l, CategoroidsVocabularyPtr coid_vocab)
  {
    // создаём справочник словосочетаний
    std::shared_ptr< MweVocabulary > v_mwe = std::make_shared<MweVocabulary>();
    if ( !v_mwe->load(mwe_fn) )
      return false;
    // открываем файл с тренировочными данными
    FILE *conll_file = fopen(conll_fn.c_str(), "rb");
    if ( conll_file == nullptr )
    {
      std::cerr << "Train-file open: error: " << std::strerror(errno) << std::endl;
      return false;
    }
    // создаем контейнер для словаря
    VocabMappingPtr vocab_lemma = std::make_shared<VocabMapping>();
    // в цикле читаем предложения из CoNLL-файла и извлекаем из них информацию для словаря
    SentenceMatrix sentence_matrix;
    sentence_matrix.reserve(5000);
    StatHelper stat;
    while ( !feof(conll_file) )
    {
      bool succ = ConllReader::read_sentence(conll_file, sentence_matrix);
      stat.calc_sentence(sentence_matrix.size());
      if (!succ)
      {
        stat.inc_sr_fils();
        continue;
      }
      if (sentence_matrix.size() == 0)
        continue;
      apply_patches(sentence_matrix); // todo: УБРАТЬ!  временный дополнительный корректор для борьбы с "грязными данными" в результатах лемматизации
      v_mwe->put_phrases_into_sentence(sentence_matrix);
      process_sentence_lemmas(vocab_lemma, sentence_matrix);
    }
    fclose(conll_file);
    std::cout << std::endl;
    stat.output_stat();
    // сохраняем словарь в файл
    std::cout << "Save lemmas main vocabulary..." << std::endl;
    reduce_vocab(vocab_lemma, limit_l, coid_vocab);
    save_vocab(vocab_lemma, voc_l_fn);
    return true;
  } // method-end
  // проверка, является ли токен числовой конструкцией
  bool isNumeric(const std::string& lemma)
  {
    return ( lemma == "@num@" || lemma == "@num@,@num@" || lemma == "@num@:@num@" ||
             lemma == "@num@-@num@" || lemma == "@num@--@num@" || lemma == "@num@‒@num@" || lemma == "@num@–@num@" || lemma == "@num@—@num@" );
  } // method-end
//  // проверка, является ли токен стоп-словом для основного словаря
//  void erase_main_stopwords(VocabMappingPtr vocab)
//  {
//    static bool isListLoaded = false;
//    std::set<std::string> stoplist;
//    if (!isListLoaded)
//    {
//      isListLoaded = true;
//      std::ifstream ifs("stopwords.lems");
//      std::string line;
//      while ( std::getline(ifs, line).good() )
//        stoplist.insert(line);
//    }
//    std::cout << "  stopwords reduce (main)" << std::endl;
//    auto it = vocab->begin();
//    while (it != vocab->end())    //TODO: в C++20 заменить на std::erase_if (https://en.cppreference.com/w/cpp/container/map/erase_if)
//    {
//      if (stoplist.find(it->first) == stoplist.end())
//        ++it;
//      else
//        it = vocab->erase(it);
//    }
//  } // method-end
  // проверка, является ли токен стоп-словом для словаря токенов (служит для исправления ошибок токенизации)
  void erase_toks_stopwords(VocabMappingPtr vocab)
  {
    static bool isListLoaded = false;
    std::set<std::string> stoplist;
    if (!isListLoaded)
    {
      isListLoaded = true;
      std::ifstream ifs("stopwords.toks");
      std::string line;
      while ( std::getline(ifs, line).good() )
        stoplist.insert(line);
    }
    std::cout << "  stopwords reduce (toks)" << std::endl;
    auto it = vocab->begin();
    while (it != vocab->end())    //TODO: в C++20 заменить на std::erase_if (https://en.cppreference.com/w/cpp/container/map/erase_if)
    {
      if (stoplist.find(it->first) == stoplist.end())
        ++it;
      else
        it = vocab->erase(it);
    }
  } // method-end
  // удаление OOV-суффиксов, встречающихся при небольшом числе слов
  void oov_idf_filter(VocabMappingPtr oov_vocab, VocabMappingPtr tokens_vocab, size_t max_oov_sfx)
  {
    // фильтрующий предел (суффикс должен встречаться в таком или более числе слов)
    const size_t REP_LIM = 50;
    // строим из токенов отображение: суффикс -> в скольких разных словах он встречается
    std::map<std::string, size_t> sfx2wc;
    std::set<std::string> good_sfxs;
    for ( auto& vmi : (*tokens_vocab) )
    {
      auto word = StrConv::To_UTF32(vmi.first);
      auto wl = word.length();
      if (wl < SFX_SOURCE_WORD_MIN_LEN)
        continue;
      std::string sfx;
      for (size_t i = 0; i < max_oov_sfx; ++i)
      {
        if (wl <= i)
          break;
        sfx = StrConv::To_UTF8(std::u32string(1, word[wl-i-1])) + sfx;
        sfx2wc[sfx] += 1;
        if (sfx2wc[sfx] == REP_LIM)
          good_sfxs.insert(OOV+sfx);
      }
    }
    // фильтруем построенный oov_vocab
    std::cout << "  representativeness reduce (oov)" << std::endl;
    auto it = oov_vocab->begin();
    while (it != oov_vocab->end())    //TODO: в C++20 заменить на std::erase_if (https://en.cppreference.com/w/cpp/container/map/erase_if)
    {
      if (good_sfxs.find(it->first) != good_sfxs.end())
        ++it;
      else
        it = oov_vocab->erase(it);
    }
  } // method-end
  bool is_punct__patch(const std::string& word)
  {
    const std::set<std::string> puncts = { ".", ",", "!", "?", ":", ";", "…", "...", "--", "—", "–", "‒",
                                           "'", "ʼ", "ˮ", "\"", "«", "»", "“", "”", "„", "‟", "‘", "’", "‚", "‛",
                                           "(", ")", "[", "]", "{", "}", "⟨", "⟩" };
    if ( puncts.find(word) != puncts.end() )
      return true;
    else
      return false;
  }
  void apply_patches(SentenceMatrix& sentence)
  {
    for ( auto& token : sentence )
      if ( is_punct__patch(token[Conll::LEMMA]) )
        token[Conll::DEPREL] = "PUNC";
  } // method-end
  void process_sentence_lemmas(VocabMappingPtr vocab, const SentenceMatrix& sentence)
  {
    for ( auto& token : sentence )
    {
      if (token[Conll::DEPREL] == "PUNC")  // знаки препинания в основной словарь не включаем (они обрабатываются особо)
        continue;
      if ( token[Conll::LEMMA] == "_" ) // символ отсутствия значения в conll
        continue;
      auto& word = token[Conll::LEMMA];
      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;
    }
  } // method-end
  void process_sentence_tokens(VocabMappingPtr vocab, Token2LemmasMapPtr token2lemmas_map, /*bool excludeNumsFromToks,*/ const SentenceMatrix& sentence)
  {
    for ( auto& token : sentence )
    {
      if (token[Conll::DEPREL] == "PUNC")  // знаки препинания в словарь не включаем (они обрабатываются особо)
        continue;
      if ( token[Conll::FORM] == "_" || token[Conll::LEMMA] == "_" )   // символ отсутствия значения в conll
        continue;
//      if ( excludeNumsFromToks && isNumeric(token[Conll::LEMMA]) )          // @num@:@num@ и т.п.
//        continue;
      auto word = token[Conll::FORM];

      auto it = vocab->find( word );
      if (it == vocab->end())
        (*vocab)[word] = 1;
      else
        ++it->second;

      auto itt = (*token2lemmas_map)[word].find( token[Conll::LEMMA] );
      if ( itt == (*token2lemmas_map)[word].end() )
        (*token2lemmas_map)[word][token[Conll::LEMMA]] = 1;
      else
        ++itt->second;
    }
  } // method-end
  void process_sentence_oov(VocabMappingPtr vocab, const SentenceMatrix& sentence, size_t max_oov_sfx)
  {
    for ( auto& token : sentence )
    {
      if (token[Conll::DEPREL] == "PUNC")  // знаки препинания в словарь не включаем (они обрабатываются особо)
        continue;
      if ( token[Conll::FORM] == "_" || token[Conll::LEMMA] == "_" )   // символ отсутствия значения в conll
        continue;
      if ( isNumeric(token[Conll::LEMMA]) )
        continue;
      auto word = StrConv::To_UTF32(token[Conll::FORM]);
      auto wl = word.length();

      const std::u32string Digs = U"0123456789";
      const std::u32string RuLets = U"АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя";

      // вариант "первая буква, последняя цифра"
      if ( RuLets.find(word.front()) != std::u32string::npos && Digs.find(word.back()) != std::u32string::npos )
      {
        (*vocab)[OOV+"LD_"] += 1;
        continue;
      }
      // вариант "кириллический суффикс"
      if (wl < SFX_SOURCE_WORD_MIN_LEN)
        continue;
      std::string sfx;
      for (size_t i = 0; i < max_oov_sfx; ++i)
      {
        if (wl <= i)
          break;
        std::u32string::value_type letter = word[wl-i-1];
        bool isCyr = (RuLets.find(letter) != std::u32string::npos);
        if (letter == U'-' && (sfx == "то" || sfx == "ка"))
          isCyr = true;
        if (!isCyr)
          break;
        sfx = StrConv::To_UTF8(std::u32string(1, letter)) + sfx;
        (*vocab)[OOV+sfx] += 1;
      }
    } // for all tokens in sentence
  } // method-end
  void process_sentence_dep_ctx(VocabMappingPtr vocab, const SentenceMatrix& sentence, size_t column, bool use_deprel)
  {
    for (auto& token : sentence)
    {
      if ( use_deprel )
      {
        if ( token[Conll::DEPREL] == "PUNC" )  // знаки препинания в словарь синтаксических контекстов не включаем
          continue;
        if ( token[column] == "_" || token[Conll::DEPREL] == "_" )  // символ отсутствия значения в conll
          continue;
        size_t parent_token_no = 0;
        try {
          parent_token_no = std::stoi(token[Conll::HEAD]);
        } catch (...) {
          parent_token_no = 0; // если конвертирование неудачно, считаем, что нет родителя
        }
        if ( parent_token_no == 0 )
          continue;
        auto& parent = sentence[ parent_token_no - 1 ];
        if ( parent[7] == "PUNC" ) // "контексты -- знаки препинания" нам не интересны
          continue;                // note: не посчитаем контекст вниз, но его и не нужно, т.к. это контекст знака пунктуации
        if ( parent[column] == "_" ) // символ отсутствия значения в conll
          continue;

        // рассматриваем контекст с точки зрения родителя в синтаксической связи
        auto ctx__from_head_viewpoint = token[column] + "<" + token[Conll::DEPREL];
        auto it_h = vocab->find( ctx__from_head_viewpoint );
        if (it_h == vocab->end())
          (*vocab)[ctx__from_head_viewpoint] = 1;
        else
          ++it_h->second;
        // рассматриваем контекст с точки зрения потомка в синтаксической связи
        auto ctx__from_child_viewpoint = parent[column] + ">" + token[Conll::DEPREL];
        auto it_c = vocab->find( ctx__from_child_viewpoint );
        if (it_c == vocab->end())
          (*vocab)[ctx__from_child_viewpoint] = 1;
        else
          ++it_c->second;
      }
      else
      {
        if ( token[Conll::DEPREL] == "PUNC" )   // знаки препинания в словарь синтаксических контекстов не включаем
          continue;
        if ( token[column] == "_" ) // символ отсутствия значения в conll
          continue;
        auto& word = token[column];
        auto it = vocab->find( word );
        if (it == vocab->end())
          (*vocab)[word] = 1;
        else
          ++it->second;
      } // if ( use_depre ) then ... else ...
    }
  } // method-end
  // редукция и сохранение словаря в файл
  void save_vocab(VocabMappingPtr vocab, const std::string& file_name, Token2LemmasMapPtr t2l = nullptr, const std::string& tlm_fn = std::string())
  {
    // пересортируем в порядке убывания частоты
    std::multimap<uint64_t, std::string, std::greater<uint64_t>> revVocab;
    for (auto& record : *vocab)
      revVocab.insert( std::make_pair(record.second, record.first) );
    // сохраняем словарь в файл
    FILE *fo = fopen(file_name.c_str(), "wb");
    for (auto& record : revVocab)
      fprintf(fo, "%s %lu\n", record.second.c_str(), record.first);
    fclose(fo);
    // сохраняем мэппинг из токенов в леммы (если в функцию передана соответствующая структура)
    // сохранение происходит с учётом отсечения по частотному порогу и переупорядочения (как в самом словаре токенов)
    if (t2l)
    {
      std::ofstream t2l_fs( tlm_fn.c_str() );
      for (auto& record : revVocab)
      {
        auto it = t2l->find(record.second);
        if (it == t2l->end()) continue;
        t2l_fs << record.second;
        for (auto& lemma : it->second)
          t2l_fs << " " << lemma.first << " " << lemma.second;
        t2l_fs << "\n";
      }
    }
  } // method-end
  // редукция словаря по частотному порогу
  void reduce_vocab(VocabMappingPtr vocab, size_t min_count, CategoroidsVocabularyPtr coid_vocab = nullptr)
  {
    // удаляем редкие слова (ниже порога отсечения)
    std::cout << "  min-count reduce" << std::endl;
    auto it = vocab->begin();
    while (it != vocab->end())    //TODO: в C++20 заменить на std::erase_if (https://en.cppreference.com/w/cpp/container/map/erase_if)
    {
      if (it->second >= min_count)
        ++it;
      else
      {
        if (coid_vocab && it->second >= 5 && coid_vocab->in_words_list(it->first))
          ++it;
        else
          it = vocab->erase(it);
      }
    }
    std::cout << "  resulting vocabulary size: " << vocab->size() << std::endl;
  }
};


#endif /* VOCABS_BUILDER_H_ */
