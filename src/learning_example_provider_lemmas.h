#ifndef LEARNING_EXAMPLE_PROVIDER_LEMMAS_H_
#define LEARNING_EXAMPLE_PROVIDER_LEMMAS_H_

#include "learning_example_provider.h"
#include "mwe_vocabulary.h"
#include "external_vocabs_manager.h"

#include <memory>
#include <vector>
#include <optional>
#include <cstring>       // for std::strerror
#include <cmath>

//#include "log.h"


// Класс поставщика обучающих примеров ("итератор" по обучающему множеству) для обучения категориальных и тематических эмбеддингов.
class LearningExampleProviderLemmas : public LearningExampleProvider
{
public:
  // конструктор
  LearningExampleProviderLemmas(const CommandLineParametersDefs& cmdLineParams,
                                std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary,
                                bool trainTokens,
                                std::shared_ptr<OriginalWord2VecVocabulary> depCtxVocabulary, std::shared_ptr<OriginalWord2VecVocabulary> assocCtxVocabulary,
                                std::shared_ptr<MweVocabulary> mweVocabulary,
                                size_t embColumn,
                                std::shared_ptr< ExternalVocabsManager > ext_vm = nullptr)
  : LearningExampleProvider(cmdLineParams, wordsVocabulary)
  , toks_train(trainTokens)
  , dep_ctx_vocabulary(depCtxVocabulary)
  , assoc_ctx_vocabulary(assocCtxVocabulary)
  , mwe_vocabulary(mweVocabulary)
  , emb_column(embColumn)
  , dep_column( cmdLineParams.getAsInt("-col_ctx_d") - 1 )
  , use_deprel( cmdLineParams.getAsInt("-use_deprel") == 1 )
  , sample_d( cmdLineParams.getAsFloat("-sample_d") )
  , sample_a( cmdLineParams.getAsFloat("-sample_a") )
  , ext_vocabs_manager(ext_vm)
  {
    if ( dep_ctx_vocabulary )
      dep_ctx_vocabulary->sampling_estimation(sample_d);
    if ( assoc_ctx_vocabulary )
      assoc_ctx_vocabulary->sampling_estimation(sample_a);
  } // constructor-end

  // извлечение обучающих примеров из предложения (вспомогат. процедура для get)
  virtual void get_lp_specific(ThreadEnvironment& t_environment, float fraction)
  {
    auto& sentence_matrix = t_environment.sentence_matrix;

    // добавим в предложение фразы (преобразуя sentence_matrix)
    if (mwe_vocabulary)
    {
      mwe_vocabulary->put_phrases_into_sentence(sentence_matrix);
    }

    auto sm_size = sentence_matrix.size();
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    // конвертируем conll-таблицу в более удобные структуры
    const auto synt_related = LearningExampleProvider::get_syntactically_related(sentence_matrix);
    std::vector< std::vector<size_t> > deps( sm_size );      // хранилище синатксических контекстов для каждого токена
    std::vector< std::vector<size_t> > assocs( sm_size );    // хранилище ассоциативных контекстов для каждого токена
    if ( dep_ctx_vocabulary )
    {
      for (size_t i = 0; i < sm_size; ++i)
      {
        auto& token = sentence_matrix[i];
        if ( token[Conll::DEPREL] == "_" ) continue;
        size_t parent_token_no = 0;
        try {
          parent_token_no = std::stoi(token[Conll::HEAD]);
        } catch (...) {
          parent_token_no = 0; // если конвертирование неудачно, считаем, что нет родителя
        }
        if ( parent_token_no < 1 || parent_token_no > sm_size ) continue;

        // рассматриваем контекст с точки зрения родителя в синтаксической связи
        auto ctx__from_head_viewpoint = ( use_deprel ? token[dep_column] + "<" + token[Conll::DEPREL] : token[dep_column] );
        auto ctx__fhvp_idx = dep_ctx_vocabulary->word_to_idx( ctx__from_head_viewpoint );
        if ( ctx__fhvp_idx != INVALID_IDX )
          deps[ parent_token_no - 1 ].push_back( ctx__fhvp_idx );
        // рассматриваем контекст с точки зрения потомка в синтаксической связи
        auto& parent = sentence_matrix[ parent_token_no - 1 ];
        auto ctx__from_child_viewpoint = (use_deprel ? parent[dep_column] + ">" + token[Conll::DEPREL] : parent[dep_column] );
        auto ctx__fcvp_idx = dep_ctx_vocabulary->word_to_idx( ctx__from_child_viewpoint );
        if ( ctx__fcvp_idx != INVALID_IDX )
          deps[ i ].push_back( ctx__fcvp_idx );
      }
      if (sample_d > 0)
      {
        for (size_t i = 0; i < sm_size; ++i)
        {
          auto tdcIt = deps[i].begin();
          while (tdcIt != deps[i].end())
          {
            float ran = dep_ctx_vocabulary->idx_to_data(*tdcIt).sample_probability;
            t_environment.update_random();
            if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
              tdcIt = deps[i].erase(tdcIt);
            else
              ++tdcIt;
          }
        }
      }
    }
    if ( assoc_ctx_vocabulary && sm_size > 20 /* на коротких предложениях ассоциацию не учим вообще */)
    {
      // первичная фильтрация (сабсэмплинг и несловарное)
      std::vector<std::optional<size_t>> associations(sm_size);
      for (size_t i = 0; i < sm_size; ++i)
      {
        auto& token = sentence_matrix[i];
        // обязательно проверяем по словарю ассоциаций, т.к. он фильтруется (в отличие от главного)
        // но индекс берем из главного (т.к. основной алгортим работает только по первой матрице)
        std::string avi = (!toks_train) ? token[Conll::LEMMA] : token[Conll::FORM];    // lemma column by default; lower(token) when token training
        size_t assoc_idx = assoc_ctx_vocabulary->word_to_idx(avi);
        if ( assoc_idx == INVALID_IDX )
          continue;
        // применяем сабсэмплинг к ассоциациям
        if (sample_a > 0)
        {
          float ran = assoc_ctx_vocabulary->idx_to_data(assoc_idx).sample_probability;
          t_environment.update_random();
          if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
            continue;
        }
        auto word_idx = words_vocabulary->word_to_idx(token[emb_column]);
        if ( word_idx == INVALID_IDX )
          continue;
        associations[i] = word_idx;
      } // for all words in sentence
      // формирование ассоциативных контекстов с фильтрацией синтаксически-связанных
      for (size_t i = 0; i < sm_size; ++i)
      {
        if ( !associations[i] ) continue; // i-ое слово отфильтровано
        for (size_t j = 0; j < sm_size; ++j)
        {
          if ( j == i ) continue; // сам себе не ассоциативен (бессмысленные вычисления)
          if ( associations[j] && synt_related[i].find(j) == synt_related[i].end() ) // если j-ое слово не отфильтровано и не является синтаксически свяазнным в i-ым
            assocs[i].push_back(associations[j].value());                              // то добавим его в ассоциации к i-ому
        }
      }
// // DEBUG
// for (size_t i = 0; i < sm_size; ++i) {
//   std::cout << (i+1) << " " << sentence_matrix[i][Conll::FORM] << " / ";
//   for (const auto& j : synt_related[i]) std::cout << (j+1) << " ";
//   std::cout << "/ ";
//   if ( associations[i] ) std::cout << associations[i].value(); else std::cout << "-";
//   std::cout << " / ";
//   for (const auto& j : assocs[i]) std::cout << j << " ";
//   std::cout << std::endl;
// }
//   int uuuu = 0;
//   std::cin >> uuuu;
// // DEBUG
    }
    // конвертируем в структуру для итерирования (фильтрация несловарных)
    for (size_t i = 0; i < sm_size; ++i)
    {
      auto word_idx = words_vocabulary->word_to_idx(sentence_matrix[i][emb_column]);
      if ( word_idx != INVALID_IDX )
      {
        ++t_environment.words_count;
        if (sample_w > 0)
        {
          float ran = words_vocabulary->idx_to_data(word_idx).sample_probability;
          t_environment.update_random();
          if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
            continue;
        }
        LearningExample le;
        le.word = word_idx;
        le.dep_context = deps[i];
        le.assoc_context = assocs[i];

        if (ext_vocabs_manager)
        {
          ext_vocabs_manager->get(le.ext_vocab_data, fraction, t_environment.next_random);
        }

        t_environment.sentence.push_back(le);
      }
    }
  } // method-end

  // изменение subsampling-коэффициентов в динамике
  virtual void update_subsampling_rates(float w_mul = 0.7 /* , float d_mul = 0.95, float a_mul = 0.95*/)
  {
    sample_w *= w_mul; /*sample_d *= d_mul; sample_a *= a_mul;*/
    if ( words_vocabulary )
      words_vocabulary->sampling_estimation(sample_w);
    // if ( dep_ctx_vocabulary )
    //   dep_ctx_vocabulary->sampling_estimation(sample_d);
    // if ( assoc_ctx_vocabulary )
    //   assoc_ctx_vocabulary->sampling_estimation(sample_a);
  }


private:
  // словари
  bool toks_train;    // признак того, что тренируются словоформы
  std::shared_ptr< OriginalWord2VecVocabulary > dep_ctx_vocabulary;
  std::shared_ptr< OriginalWord2VecVocabulary > assoc_ctx_vocabulary;
  std::shared_ptr< MweVocabulary > mwe_vocabulary;
  // номера колонок в conll, откуда считывать данные
  size_t emb_column;
  size_t dep_column;
  // следует ли задействовать тип и направление синтаксической связи в определении синтаксического контекста
  bool use_deprel;
  // порог для алгоритма сэмплирования (subsampling) -- для синтаксических контекстов
  float sample_d = 0;
  // порог для алгоритма сэмплирования (subsampling) -- для ассоциативных контекстов
  float sample_a = 0;
  // менеджер внешних словарей
  std::shared_ptr< ExternalVocabsManager > ext_vocabs_manager;

}; // class-decl-end


#endif /* LEARNING_EXAMPLE_PROVIDER_LEMMAS_H_ */
