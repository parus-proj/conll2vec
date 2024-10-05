#ifndef LEARNING_EXAMPLE_PROVIDER_GRAMM_H_
#define LEARNING_EXAMPLE_PROVIDER_GRAMM_H_

#include "learning_example_provider.h"

#include <memory>
#include <vector>
#include <optional>
#include <cstring>       // for std::strerror
#include <cmath>

//#include "log.h"


// Класс поставщика обучающих примеров ("итератор" по обучающему множеству) для обучения грамматических эмбеддингов.
class LearningExampleProviderGramm : public LearningExampleProvider
{
public:

  // конструктор
  LearningExampleProviderGramm( const CommandLineParametersDefs& cmdLineParams,
                                std::shared_ptr<OriginalWord2VecVocabulary> wordsVocabulary,
                                bool oov, 
                                size_t oovMaxLen 
                              )
  : LearningExampleProvider(cmdLineParams, wordsVocabulary)
  , train_oov(oov)
  , max_oov_sfx(oovMaxLen)
  {
  } // constructor-end

  // извлечение из предложения обучающих примеров для построения грамматических векторов (вспомогат. процедура для get в родительском классе)
  virtual void get_lp_specific(ThreadEnvironment& t_environment, float /*fraction*/)
  {
    auto& sentence_matrix = t_environment.sentence_matrix;
    auto sm_size = sentence_matrix.size();
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < sm_size; ++i)
    {
      if (sentence_matrix[i][Conll::LEMMA] == "_") // если процедура fit по какой-то причине сделала лемму невалидной, то надо это пропустить
        continue;
      auto token_str = sentence_matrix[i][Conll::FORM];
      auto word_idx = words_vocabulary->word_to_idx(token_str);
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
        msd2vec(le, sentence_matrix[i][Conll::FEATURES]);  // конструирование грамматического вектора из набора граммем
        t_environment.sentence.push_back(le);
      }
      if (train_oov)
        try_to_get_oov_suffixes(t_environment, token_str, sentence_matrix[i][Conll::FEATURES]);
    }
  } // method-end

  void try_to_get_oov_suffixes(ThreadEnvironment& t_environment, const std::string& token, const std::string& msd)
  {
    const std::string OOV = "_OOV_";
    const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
    auto t32 = StrConv::To_UTF32(token);
    auto wl = t32.length();
    if (wl < SFX_SOURCE_WORD_MIN_LEN)
      return;
    std::string sfx;
    for (size_t i = 0; i < max_oov_sfx; ++i)
    {
      if (wl <= i)
        break;
      sfx = StrConv::To_UTF8(std::u32string(1, t32[wl-i-1])) + sfx;
      auto oov_str = OOV+sfx;
      auto word_idx = words_vocabulary->word_to_idx(oov_str);
      if ( word_idx != INVALID_IDX )
      {
        ++t_environment.words_count;
        if (sample_w > 0)
        {
          float ran = words_vocabulary->idx_to_data(word_idx).sample_probability;
          t_environment.update_random();
          if (ran < (t_environment.next_random & 0xFFFF) / (float)65536)
            continue; // попробуем другие суффиксы, этот пропустим
        }
        LearningExample le;
        le.word = word_idx;
        msd2vec(le, msd);  // конструирование грамматического вектора из набора граммем
        t_environment.sentence.push_back(le);
      }
    }
  } // method-end

  // получение длины вектора граммем
  size_t getGrammemesVectorSize() const
  {
    return gcLast;
  }
private:
  // нужно ли обучать oov-суффиксы
  bool train_oov;
  // максимальная длина oov-суффикса
  size_t max_oov_sfx;
  // минимальная длина слова, от которого берутся oov-суффиксы
  const size_t SFX_SOURCE_WORD_MIN_LEN = 6;

private:
  enum GramCode2VecPosition
  {
    gcPosNoun = 0,
    gcPosPron,
    gcPosAdj,
    gcPosNumeral,
    gcPosAdv,
    gcPosVerb,
    gcPosAdpos,
    gcPosConj,
    gcPosPart,
    gcPosInter,
    gcNtCmn,
    gcNtProper,
    gcGendMas,
    gcGendFem,
    gcGendNeu,
    gcNumSing,
    gcNumPlur,
    gcCaseNom,
    gcCaseGen,
    gcCaseDat,
    gcCaseAcc,
    gcCaseIns,
    gcCaseLoc,
    gcCaseVoc,
    gcAnimYes,
    gcAnimNo,
    gcDegrPos,
    gcDegrCom,
    gcDegrSup,
    gcDefShort,
    gcDefFull,
    gcPossess,
    gcVfInd,
    gcVfImp,
    gcVfCond,
    gcVfInf,
    gcVfPart,
    gcVfGer,
    gcTensePre,
    gcTensePast,
    gcTenseFut,
    gcPers1,
    gcPers2,
    gcPers3,
    gcVoiceAct,
    gcVoicePass,
    gcAspProg,
    gcAspPerf,
    gcPrntPers,
    gcPrntDem,
    gcPrntIndef,
    gcPrntInterrog,
    gcPrntRelat,
    gcPrntReflex,
    gcPrntNeg,
    gcPrntNspec,
    gcStNom,
    gcStAdj,
    gcStAdv,
    gcNumeralCard,
    gcNumeralOrd,
    gcNumeralCollect,
    gcCtCoord,
    gcCtSubord,

    gcLast
  };
  void encodeGender(char value, LearningExample& le)
  {
    switch(value)
    {
    case 'm': le.assoc_context[gcGendMas] = 1; break;
    case 'f': le.assoc_context[gcGendFem] = 1; break;
    case 'n': le.assoc_context[gcGendNeu] = 1; break;
    }
  }
  void encodeNumber(char value, LearningExample& le)
  {
    switch(value)
    {
    case 's': le.assoc_context[gcNumSing] = 1; break;
    case 'p': le.assoc_context[gcNumPlur] = 1; break;
    }
  }
  void encodeCase(char value, LearningExample& le)
  {
    switch (value)
    {
    case 'n': le.assoc_context[gcCaseNom] = 1; break;
    case 'g': le.assoc_context[gcCaseGen] = 1; break;
    case 'd': le.assoc_context[gcCaseDat] = 1; break;
    case 'a': le.assoc_context[gcCaseAcc] = 1; break;
    case 'i': le.assoc_context[gcCaseIns] = 1; break;
    case 'l': le.assoc_context[gcCaseLoc] = 1; break;
    case 'v': le.assoc_context[gcCaseVoc] = 1; break;
    }
  }
  void encodeAnim(char value, LearningExample& le)
  {
    switch(value)
    {
    case 'y': le.assoc_context[gcAnimYes] = 1; break;
    case 'n': le.assoc_context[gcAnimNo] = 1; break;
    }
  }
  void encodeTense(char value, LearningExample& le)
  {
    switch(value)
    {
    case 'p': le.assoc_context[gcTensePre] = 1; break;
    case 'f': le.assoc_context[gcTenseFut] = 1; break;
    case 's': le.assoc_context[gcTensePast] = 1; break;
    }
  }
  void encodePerson(char value, LearningExample& le)
  {
    switch(value)
    {
    case '1': le.assoc_context[gcPers1] = 1; break;
    case '2': le.assoc_context[gcPers2] = 1; break;
    case '3': le.assoc_context[gcPers3] = 1; break;
    }
  }
  void encodeDefiniteness(char value, LearningExample& le)
  {
    switch(value)
    {
    case 's': le.assoc_context[gcDefShort] = 1; break;
    case 'f': le.assoc_context[gcDefFull] = 1; break;
    }
  }
  void encodeDegree(char value, LearningExample& le)
  {
    switch(value)
    {
    case 'p': le.assoc_context[gcDegrPos] = 1; break;
    case 'c': le.assoc_context[gcDegrCom] = 1; break;
    case 's': le.assoc_context[gcDegrSup] = 1; break;
    }
  }
  void msd2vec(LearningExample& le, const std::string& msd)
  {
    le.assoc_context.resize(gcLast, 0);
    if (msd.empty()) return;
    if (msd[0] == 'N') // noun
    {
      le.assoc_context[gcPosNoun] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1 && msd[i] == 'c') le.assoc_context[gcNtCmn] = 1;
        if (i == 1 && msd[i] == 'p') le.assoc_context[gcNtProper] = 1;
        if (i == 2) encodeGender(msd[i], le);
        if (i == 3) encodeNumber(msd[i], le);
        if (i == 4) encodeCase(msd[i], le);
        if (i == 5) encodeAnim(msd[i], le);
      }
    }
    if (msd[0] == 'V') // verb
    {
      le.assoc_context[gcPosVerb] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 2)
        {
          switch (msd[i])
          {
          case 'i': le.assoc_context[gcVfInd] = 1; break;
          case 'm': le.assoc_context[gcVfImp] = 1; break;
          case 'c': le.assoc_context[gcVfCond] = 1; break;
          case 'n': le.assoc_context[gcVfInf] = 1; break;
          case 'p': le.assoc_context[gcVfPart] = 1; break;
          case 'g': le.assoc_context[gcVfGer] = 1; break;
          }
        }
        if (i == 3) encodeTense(msd[i], le);
        if (i == 4) encodePerson(msd[i], le);
        if (i == 5) encodeNumber(msd[i], le);
        if (i == 6) encodeGender(msd[i], le);
        if (i == 7)
        {
          switch (msd[i])
          {
          case 'a': le.assoc_context[gcVoiceAct] = 1; break;
          case 'p': le.assoc_context[gcVoicePass] = 1; break;
          }
        }
        if (i == 8) encodeDefiniteness(msd[i], le);
        if (i == 9)
        {
          switch (msd[i])
          {
          case 'p': le.assoc_context[gcAspProg] = 1; break;
          case 'e': le.assoc_context[gcAspPerf] = 1; break;
          }
        }
        if (i == 10) encodeCase(msd[i], le);
      }
    }
    if (msd[0] == 'A') // adjective
    {
      le.assoc_context[gcPosAdj] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1 and msd[i] == 's') le.assoc_context[gcPossess] = 1;
        if (i == 2) encodeDegree(msd[i], le);
        if (i == 3) encodeGender(msd[i], le);
        if (i == 4) encodeNumber(msd[i], le);
        if (i == 5) encodeCase(msd[i], le);
        if (i == 6) encodeDefiniteness(msd[i], le);
      }
    }
    if (msd[0] == 'P') // pronoun
    {
      le.assoc_context[gcPosPron] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1)
        {
          switch (msd[i])
          {
          case 'p': le.assoc_context[gcPrntPers] = 1; break;
          case 'd': le.assoc_context[gcPrntDem] = 1; break;
          case 'i': le.assoc_context[gcPrntIndef] = 1; break;
          case 's': le.assoc_context[gcPossess] = 1; break;
          case 'q': le.assoc_context[gcPrntInterrog] = 1; break;
          case 'r': le.assoc_context[gcPrntRelat] = 1; break;
          case 'x': le.assoc_context[gcPrntReflex] = 1; break;
          case 'z': le.assoc_context[gcPrntNeg] = 1; break;
          case 'n': le.assoc_context[gcPrntNspec] = 1; break;
          }
        }
        if (i == 2) encodePerson(msd[i], le);
        if (i == 3) encodeGender(msd[i], le);
        if (i == 4) encodeNumber(msd[i], le);
        if (i == 5) encodeCase(msd[i], le);
        if (i == 6)
        {
          switch (msd[i]) // todo: попробовать перекодировать в части речи
          {
          case 'n': le.assoc_context[gcStNom] = 1; break;
          case 'a': le.assoc_context[gcStAdj] = 1; break;
          case 'r': le.assoc_context[gcStAdv] = 1; break;
          }
        }
        if (i == 7) encodeAnim(msd[i], le);
      }
    }
    if (msd[0] == 'R') // adverb
    {
      le.assoc_context[gcPosAdv] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1) encodeDegree(msd[i], le);
      }
    }
    if (msd[0] == 'M') // numeral
    {
      le.assoc_context[gcPosNumeral] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1)
        {
          switch (msd[i])
          {
          case 'c': le.assoc_context[gcNumeralCard] = 1; break;
          case 'o': le.assoc_context[gcNumeralOrd] = 1; break;
          case 'l': le.assoc_context[gcNumeralCollect] = 1; break;
          }
        }
        if (i == 2) encodeGender(msd[i], le);
        if (i == 3) encodeNumber(msd[i], le);
        if (i == 4) encodeCase(msd[i], le);
      }
    }
    if (msd[0] == 'S') // adposition
    {
      le.assoc_context[gcPosAdpos] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 3) encodeCase(msd[i], le);
      }
    }
    if (msd[0] == 'C') // conjunction
    {
      le.assoc_context[gcPosConj] = 1;
      for(size_t i = 1; i < msd.size(); ++i)
      {
        if (i == 1)
        {
          switch (msd[i])
          {
          case 'c': le.assoc_context[gcCtCoord] = 1; break;
          case 's': le.assoc_context[gcCtSubord] = 1; break;
          }
        }
      }
    }
    if (msd[0] == 'Q') // particle
      le.assoc_context[gcPosPart] = 1;
    if (msd[0] == 'I') // interjection
      le.assoc_context[gcPosInter] = 1;
  } // method-end

}; // class-decl-end


#endif /* LEARNING_EXAMPLE_PROVIDER_GRAMM_H_ */
