#ifndef ADD_PUNCT_H_
#define ADD_PUNCT_H_

#include "vectors_model.h"

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

// Добавление знаков пунктуации в модель
class AddPunct
{
public:
  static void run( const std::string& model_fn )
  {
    // 1. Загружаем модель
    VectorsModel vm;
    if ( !vm.load(model_fn) )
      return;
    // 2. Расширяем модель знаками пунктуации и сохраняем её
    if ( merge_punct(vm) )
      vm.save(model_fn);
  } // method-end

  static bool merge_punct(VectorsModel& vm)
  {
    // 1. Порождаем эмбеддинги для знаков пунктуации (в отдельной модели)
    const std::vector<std::string> puncts = { ".", ",", "!", "?", ";", "…", "...",
                                              ":", "-", "--", "—", "–", "‒",
                                              "'", "ʼ", "ˮ", "\"",
                                              "«", "“", "„", "‘", "‚",
                                              "»", "”", "‟", "’", "‛",
                                              "(", "[", "{", "⟨",
                                              ")", "]", "}", "⟩"
                                             };
    VectorsModel pvm;
    pvm.words_count = puncts.size();
    pvm.emb_size = vm.emb_size;
    std::copy(puncts.begin(), puncts.end(), std::back_inserter(pvm.vocab));
    pvm.embeddings = (float *) malloc( pvm.words_count * pvm.emb_size * sizeof(float) );
    // создаём опорные эмбеддинги
    float *support_embedding = (float *) malloc(vm.emb_size*sizeof(float));
    calc_support_embedding(vm.words_count, vm.emb_size, vm.embeddings, support_embedding);
    float *dot_se      = (float *) malloc(vm.emb_size*sizeof(float));
    float *dash_se     = (float *) malloc(vm.emb_size*sizeof(float));
    float *quote_se    = (float *) malloc(vm.emb_size*sizeof(float));
    float *lquote_se   = (float *) malloc(vm.emb_size*sizeof(float));
    float *rquote_se   = (float *) malloc(vm.emb_size*sizeof(float));
    float *bracket_se  = (float *) malloc(vm.emb_size*sizeof(float));
    float *lbracket_se = (float *) malloc(vm.emb_size*sizeof(float));
    float *rbracket_se = (float *) malloc(vm.emb_size*sizeof(float));
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, support_embedding, dot_se, 7);
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, support_embedding, dash_se, 7);
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, support_embedding, quote_se, 7);
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, support_embedding, bracket_se, 7);
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, quote_se, lquote_se, 3);
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, quote_se, rquote_se, 3);
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, bracket_se, lbracket_se, 3);
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, bracket_se, rbracket_se, 3);
    // создаём эбмеддинги для знаков препинания
    for (size_t i = 0; i < puncts.size(); ++i)
    {
      const std::string& p = puncts[i];
      float* base = nullptr;
      if ( p == "." || p == "," || p == "!" || p == "?" || p == ";" || p == "…" || p == "..." )
        base = dot_se;
      else if ( p == ":" || p == "-" || p == "--" || p == "—" || p == "–" || p == "‒" )
        base = dash_se;
      else if ( p == "'" || p == "ʼ" || p == "ˮ" || p == "\"" )
        base = quote_se;
      else if ( p == "«" || p == "“" || p == "„" || p == "‘" || p == "‚" )
        base = lquote_se;
      else if ( p == "»" || p == "”" || p == "‟" || p == "’" || p == "‛" )
        base = rquote_se;
      else if ( p == "(" || p == "[" || p == "{" || p == "⟨" )
        base = lbracket_se;
      else if ( p == ")" || p == "]" || p == "}" || p == "⟩" )
        base = rbracket_se;
      VectorsModel::make_embedding_as_neighbour(vm.emb_size, base, pvm.embeddings + pvm.emb_size * i);
    }

    // 2. Расширяем модель знаками пунктуации
    return vm.merge(pvm);
  } // method-end

private:

  static void calc_support_embedding( size_t words_count, size_t emb_size, float* embeddings, float* support_embedding )
  {
    for (size_t d = 0; d < emb_size; ++d)
    {
      float rbound = -1e10;
      for (size_t w = 0; w < words_count; ++w)
      {
        float *offs = embeddings + w*emb_size + d;
        if ( *offs > rbound )
          rbound = *offs;
      }
      *(support_embedding + d) = rbound + 0.01; // добавляем немного, чтобы не растянуть пространство
    }
  } // method-end

}; // class-decl-end


#endif /* ADD_PUNCT_H_ */
