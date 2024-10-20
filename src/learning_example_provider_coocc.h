#ifndef LEARNING_EXAMPLE_PROVIDER_COOCC_H_
#define LEARNING_EXAMPLE_PROVIDER_COOCC_H_

#include "learning_example_provider.h"
#include "mwe_vocabulary.h"

#include <memory>
#include <vector>
#include <optional>
#include <cstring>       // for std::strerror
#include <cmath>

//#include "log.h"


// Класс поставщика обучающих примеров ("итератор" по обучающему множеству) для обучения эмбеддингов сочетаемости.
class LearningExampleProviderCoocc : public LearningExampleProvider
{
public:
  // конструктор
  LearningExampleProviderCoocc(const CommandLineParametersDefs& cmdLineParams,
                                std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary
                              )
  : LearningExampleProvider(cmdLineParams, wordsVocabulary)
  , sample_a( cmdLineParams.getAsFloat("-sample_a") )
  {
  } // constructor-end

  // извлечение обучающих примеров из предложения (вспомогат. процедура для get)
  virtual void get_lp_specific(ThreadEnvironment& t_environment, float fraction)
  {
    auto& sentence_matrix = t_environment.sentence_matrix;

    auto sm_size = sentence_matrix.size();
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    // конвертируем conll-таблицу в более удобные структуры
    std::vector< std::vector<size_t> > assocs( sm_size );    // хранилище сочетаемостных контекстов для каждого токена

    // первичная фильтрация (сабсэмплинг и несловарное)
    std::vector<std::optional<size_t>> associations(sm_size);
    for (size_t i = 0; i < sm_size; ++i)
    {
      auto& token = sentence_matrix[i];
      size_t assoc_idx = words_vocabulary->word_to_idx( token[Conll::FORM] );
      if ( assoc_idx == INVALID_IDX )
        continue;
      // применяем сабсэмплинг к ассоциациям
      if (sample_a > 0)
      {
        float ran = words_vocabulary->idx_to_data(assoc_idx).sample_probability;
        t_environment.update_random();
        if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
          continue;
      }
      associations[i] = assoc_idx;
    } // for all words in sentence

    t_environment.update_random();
    const float rndVal = (t_environment.next_random & 0xFFFF) / (float)65536; // случайное значение в диапазоне [0, 1]
    const bool SYNT_CTX = (rndVal > 0.6) ? true : false;

    if ( SYNT_CTX ) // 40%
    {
      // формирование ассоциативных контекстов из числа синтаксически-связанных (сильные и надежные синт.связи)
      // todo: функцию synt_related можно сделать зависимой от fraction
      const auto synt_related = LearningExampleProvider::get_syntactically_related( sentence_matrix, 
                                                                                    std::bind(&LearningExampleProviderCoocc::suitable_relation, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3) );
      for (size_t i = 0; i < sm_size; ++i)
      {
        if ( !associations[i] ) continue; // i-ое слово отфильтровано
        for (size_t j = 0; j < sm_size; ++j)
        {
          if ( j == i ) continue; // сам себе не ассоциативен (бессмысленные вычисления)
          if ( associations[j] && synt_related[i].find(j) != synt_related[i].end() ) // если j-ое слово не отфильтровано и является синтаксически свяазнным с i-ым
            assocs[i].push_back(associations[j].value());                            // то добавим его в ассоциации к i-ому
        }
      }
    }
    else
    {
      // формирование ассоциативных контекстов оконным ограничением
      t_environment.update_random();
      const float rndVal = (t_environment.next_random & 0xFFFF) / (float)65536; // случайное значение в диапазоне [0, 1]
      const size_t WINDOW_SIZE = (rndVal > 0.8) ? 2 : 1; // изредка посматриваем окно 2, чаще 1
      for (int i = 0; i < sm_size; ++i)
      {
        if ( !associations[i] ) continue; // i-ое слово отфильтровано
        int windowBegin = i - WINDOW_SIZE;
        if (windowBegin < 0) windowBegin = 0;
        int windowEnd = i + WINDOW_SIZE + 1;
        if (windowEnd > sm_size) windowEnd = sm_size;
        for (int j = windowBegin; j < windowEnd; ++j)
        {
          if ( j == i ) continue; // сам себе не ассоциативен (бессмысленные вычисления)
          if ( associations[j] ) // если j-ое слово не отфильтровано
            assocs[i].push_back(associations[j].value()); // то добавим его в ассоциации к i-ому
        }
      }
    }

    // конвертируем в структуру для итерирования (фильтрация несловарных)
    for (size_t i = 0; i < sm_size; ++i)
    {
      auto word_idx = words_vocabulary->word_to_idx(sentence_matrix[i][Conll::FORM]);
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
        le.assoc_context = assocs[i];
        t_environment.sentence.push_back(le);
      }
    }
  } // method-end

  // изменение subsampling-коэффициентов в динамике
  virtual void update_subsampling_rates(float w_mul = 0.7 /* , float a_mul = 0.95*/)
  {
    sample_w *= w_mul; /*sample_a *= a_mul;*/
    if ( words_vocabulary )
      words_vocabulary->sampling_estimation(sample_w);
    // if ( assoc_ctx_vocabulary )
    //   assoc_ctx_vocabulary->sampling_estimation(sample_a);
  }


private:
  // порог для алгоритма сэмплирования (subsampling) -- для ассоциативных контекстов
  float sample_a = 0;

  // функция фильтрации только значимых для модели совместной встречаемости синтаксических связей
  bool suitable_relation(const std::string& synt_rel_name, const std::string& child_features, const std::string& parent_features)
  {
    // return true;

    // здесь важен баланс между широтой охвата и негативным влиянием некачественной синтаксической разметки
    // будем использовать только надежные и сильные связи
    // остальное охватим оконной выборкой

    if ( child_features.empty() || parent_features.empty() || synt_rel_name.empty() ) return false; // чтобы упростить правила ниже

    // N предик V
    if ( synt_rel_name == "предик" && child_features[0] == 'N' && parent_features[0] == 'V' )
      return true;
    // N 1-компл V
    if ( synt_rel_name == "1-компл" && child_features[0] == 'N' && parent_features[0] == 'V' )
      return true;
    // N квазиагент N
    if ( synt_rel_name == "квазиагент" && child_features[0] == 'N' && parent_features[0] == 'N' )
      return true;
    // А опред N
    if ( synt_rel_name == "опред" && child_features[0] == 'A' && parent_features[0] == 'N' )
      return true;
    // R обст V
    if ( synt_rel_name == "обст" && child_features[0] == 'R' && parent_features[0] == 'V' )
      return true;

    // // P предик V
    // if ( synt_rel_name == "предик" && child_features[0] == 'P' && parent_features[0] == 'V' )
    //   return true;
    // // P 1-компл V
    // if ( synt_rel_name == "1-компл" && child_features[0] == 'P' && parent_features[0] == 'V' )
    //   return true;
    // // P квазиагент N
    // if ( synt_rel_name == "квазиагент" && child_features[0] == 'P' && parent_features[0] == 'N' )
    //   return true;

    // // Sp → ud_prepos → N
    // if ( synt_rel_name == "ud_prepos" && child_features[0] == 'S' && child_features.length()>1 && child_features[1] == 'p' && parent_features[0] == 'N')
    //   return true;


    return false;
  }

}; // class-decl-end


#endif /* LEARNING_EXAMPLE_PROVIDER_COOCC_H_ */
