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
  static void run( const std::string& model_fn, bool useTxtFmt = false )
  {
    // 1. Загружаем модель
    VectorsModel vm;
    if ( !vm.load(model_fn, useTxtFmt) )
      return;

    // 2. Добавляем в модель знаки пунктуации
    const std::set<std::string> puncts = { ".", ",", "!", "?", ":", ";", "…", "...", "--", "—", "–", "‒",
                                           "'", "ʼ", "ˮ", "\"", "«", "»", "“", "”", "„", "‟", "‘", "’", "‚", "‛",
                                           "(", ")", "[", "]", "{", "}", "⟨", "⟩" };
    for (auto p : puncts)
    {
      // проверяем наличие вектора для знака препинания в модели (если есть, то затрём его)
      size_t vec_idx = vm.get_word_idx(p);
      if (vec_idx != vm.vocab.size())
        vm.vocab[vec_idx].clear();
    }
    // создаём хранилище для новых эмбеддингов
    std::vector<std::string> new_vocab;
    float *new_embeddings = (float *) malloc( puncts.size() * vm.emb_size * sizeof(float) );
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
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dot_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back(".");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dot_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("!");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dot_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("?");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dot_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back(";");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dot_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("…");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dot_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("...");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dot_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back(",");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dash_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back(":");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dash_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("--");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dash_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("—");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dash_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("–");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, dash_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("‒");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, quote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("'");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, quote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("ʼ");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, quote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("ˮ");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, quote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("\"");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("«");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("»");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("“");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("”");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("„");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("‟");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("‘");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("’");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("‚");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rquote_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("‛");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lbracket_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("(");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rbracket_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back(")");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lbracket_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("[");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rbracket_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("]");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lbracket_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("{");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rbracket_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("}");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, lbracket_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("⟨");
    VectorsModel::make_embedding_as_neighbour(vm.emb_size, rbracket_se, new_embeddings + vm.emb_size * new_vocab.size()); new_vocab.push_back("⟩");


    // 3. Сохраняем модель, расширенную знаками пунктуации
    size_t old_vocab_size = std::count_if(vm.vocab.begin(), vm.vocab.end(), [](const std::string& item) {return !item.empty();});
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", old_vocab_size+new_vocab.size(), vm.emb_size);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
    {
      if ( vm.vocab[a].empty() )
        continue;
      VectorsModel::write_embedding(fo, useTxtFmt, vm.vocab[a], &vm.embeddings[a * vm.emb_size], vm.emb_size);
    }
    for (size_t a = 0; a < new_vocab.size(); ++a)
      VectorsModel::write_embedding(fo, useTxtFmt, new_vocab[a], &new_embeddings[a * vm.emb_size ], vm.emb_size);
    fclose(fo);

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
