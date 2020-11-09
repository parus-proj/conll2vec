#ifndef BALANCE_H_
#define BALANCE_H_

#include "vectors_model.h"

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

// Балансировщик dep/assoc-частей модели
class Balancer
{
public:
  static void run( const std::string& model_fn, bool useTxtFmt, size_t dep_size, float a_ratio )
  {
    // 1. Загружаем модель
    VectorsModel vm;
    if ( !vm.load(model_fn, useTxtFmt) )
      return;

    // 2. Сохраняем модель, корректируя веса ассоциативной части векторов
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", vm.words_count, vm.emb_size);
    for (size_t a = 0; a < vm.vocab.size(); ++a)
    {
      fprintf(fo, "%s ", vm.vocab[a].c_str());
      for (size_t b = 0; b < vm.emb_size; ++b)
      {
        float val = vm.embeddings[a * vm.emb_size + b];
        if (b >= dep_size)
          val *= a_ratio;
        if ( !useTxtFmt )
          fwrite(&val, sizeof(float), 1, fo);
        else
          fprintf(fo, " %lf", val);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);

  } // method-end
}; // class-decl-end


#endif /* BALANCE_H_ */
