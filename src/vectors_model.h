#ifndef VECTORS_MODEL_H_
#define VECTORS_MODEL_H_

#include <string>
#include <vector>
#include <fstream>
#include <iostream>


// Контейнер векторной модели вместе с функцией её загрузки в память
class VectorsModel
{
public:
  // количество слов в модели
  size_t words_count;
  // размерность пространства модели
  size_t emb_size;
  // словарь модели (имеет место соответствие между порядком слов и порядком векторов)
  std::vector<std::string> vocab;
  // векторное пространство
  float* embeddings;
public:
  // c-tor
  VectorsModel() : words_count(0), emb_size(0), embeddings(nullptr)
  {
  }
  // d-tor
  ~VectorsModel()
  {
    if (embeddings)
      free(embeddings);
  }
  // очистка модели
  void clear()
  {
    words_count = 0;
    emb_size = 0;
    vocab.clear();
    if (embeddings)
      free(embeddings);

  } // method-end
  // функция загрузки
  // выделяет память под хранение векторной модели
  bool load( const std::string& model_fn, bool useTxtFmt, bool doNormalization = false )
  {
    clear();
    // открываем файл модели
    std::ifstream ifs(model_fn.c_str(), std::ios::binary);
    if ( !ifs.good() )
    {
      std::cerr << "Model file not found" << std::endl;
      return false;
    }
    std::string buf;
    // считыавем размер матрицы
    ifs >> words_count;
    ifs >> emb_size;
    std::getline(ifs,buf); // считываем конец строки
    // выделяем память для эмбеддингов
    embeddings = (float *) malloc( words_count * emb_size * sizeof(float) );
    if (embeddings == nullptr)
    {
      std::cerr << "Cannot allocate memory: " << (words_count * emb_size * sizeof(float) / 1048576) << " MB" << std::endl;
      std::cerr << "    Words: " << words_count << std::endl;
      std::cerr << "    Embedding size: " << emb_size << std::endl;
      return false;
    }
    vocab.reserve(words_count);
    for (uint64_t w = 0; w < words_count; ++w)
    {
      std::getline(ifs, buf, ' '); // читаем слово (до пробела)
      vocab.push_back(buf);
      // читаем вектор
      float* eOffset = embeddings + w*emb_size;
      if ( !useTxtFmt )
        ifs.read( reinterpret_cast<char*>( eOffset ), sizeof(float)*emb_size );
      else
      {
        for (size_t j = 0; j < emb_size; ++j)
          ifs >> eOffset[j];
      }
      std::getline(ifs,buf); // считываем конец строки
      if ( doNormalization )
      {
        // нормируем вектор (все компоненты в диапазон [-1; +1]
        float len = std::sqrt( std::inner_product(eOffset, eOffset+emb_size, eOffset, 0.0) );
        if (len == 0)
        {
          std::cerr << "Embedding normalization error: Division by zero" << std::endl;
          clear();
          return false;
        }
        std::transform(eOffset, eOffset+emb_size, eOffset, [len](float a) -> float {return a/len;});
      }
    }
    return true;
  } // method-end
  // сохранение модели
  void save( const std::string& model_fn, bool useTxtFmt )
  {
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", words_count, emb_size);
    for (size_t a = 0; a < vocab.size(); ++a)
      VectorsModel::write_embedding(fo, useTxtFmt, vocab[a], &embeddings[a * emb_size], emb_size);
    fclose(fo);
  } // method-end
  // поиск слова в словаре
  size_t get_word_idx(const std::string& word) const
  {
    size_t widx = 0;
    for ( ; widx < vocab.size(); ++widx )
      if (vocab[widx] == word)
        break;
    return widx;
  } // method-end
  // статический метод для порождения случайного вектора, близкого к заданному (память должна быть выделена заранее)
  static void make_embedding_as_neighbour( size_t emb_size, float* base_embedding, float* new_embedding, float distance_factor = 1.0 )
  {
    auto random_sign = []() -> float { return ((float)(rand() % 2) - 0.5)/0.5; };
    for (size_t d = 0; d < emb_size; ++d)
    {
      float *offs = base_embedding + d;
      *(new_embedding + d) = *offs + random_sign() * (*offs / 100 * distance_factor);
    }
  } // method-end
  // статический метод записи одного эмбеддинга в файл
  static void write_embedding( FILE* fo, bool useTxtFmt, const std::string& word, float* embedding, size_t emb_size )
  {
    write_embedding_slice( fo, useTxtFmt, word, embedding, 0, emb_size );
  } // method-end
  static void write_embedding_slice( FILE* fo, bool useTxtFmt, const std::string& word, float* embedding, size_t begin, size_t end )
  {
    fprintf(fo, "%s", word.c_str());
    if ( !useTxtFmt )
      fprintf(fo, " ");
    for (size_t b = begin; b < end; ++b)
    {
      if ( !useTxtFmt )
        fwrite(&embedding[b], sizeof(float), 1, fo);
      else
        fprintf(fo, " %lf", embedding[b]);
    }
    fprintf(fo, "\n");
  } // method-end
}; // class-decl-end


#endif /* VECTORS_MODEL_H_ */
