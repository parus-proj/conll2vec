#ifndef VECTORS_MODEL_H_
#define VECTORS_MODEL_H_

#include <string>
#include <vector>
#include <set>
#include <map>
#include <fstream>
#include <iostream>


// Контейнер векторной модели вместе с функцией её загрузки в память
class VectorsModel
{
public:
  // количество слов в модели
  size_t words_count = 0;
  // размерность пространства модели
  size_t emb_size = 0;
  // подпространства и их размерности
  size_t dep_size = 0;
  size_t assoc_size = 0;
  size_t gramm_size = 0;
  // индексы смещений каждого из подпространств
  size_t dep_begin = 0, dep_end = 0;
  size_t assoc_begin = 0, assoc_end = 0;
  size_t gramm_begin = 0, gramm_end = 0;
  // словарь модели (имеет место соответствие между порядком слов и порядком векторов)
  std::vector<std::string> vocab;
  // векторное пространство
  float* embeddings = nullptr;
public:
  // индексы лексических единиц, не подлежащих сохранению (для упрощения логики фильтраций)
  std::set<size_t> do_not_save;
public:
  // c-tor
  VectorsModel()
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
    dep_size = 0; assoc_size = 0; gramm_size = 0;
    dep_begin = 0; dep_end = 0; assoc_begin = 0; assoc_end = 0; gramm_begin = 0; gramm_end = 0;
    vocab.clear();
    if (embeddings)
    {
      free(embeddings);
      embeddings = nullptr;
    }
    do_not_save.clear();
  } // method-end
  // инициалиация размерностей подпространств
  void setup_subspaces(size_t dep_part, size_t assoc_part, size_t gramm_part)
  {
    dep_size = dep_part;
    assoc_size = assoc_part;
    gramm_size = gramm_part;
    dep_begin = 0;
    dep_end = dep_size;
    assoc_begin = dep_end;
    assoc_end = dep_size + assoc_size;
    gramm_begin = assoc_end;
    gramm_end = dep_size + assoc_size + gramm_size;
  }
  // функция загрузки и импорта
  // выделяет память под хранение векторной модели
  bool load( const std::string& model_fn, const std::string& fmt = "c2v", bool doNormalization = false )
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
    if (fmt == "c2v")
    {
      ifs >> dep_size >> assoc_size >> gramm_size;
      setup_subspaces(dep_size, assoc_size, gramm_size);
    }
    std::getline(ifs,buf); // считываем конец строки
    // выделяем память для эмбеддингов
    embeddings = (float *) malloc( words_count * emb_size * sizeof(float) );
    if (embeddings == nullptr)
    {
      report_alloc_error();
      return false;
    }
    vocab.reserve(words_count);
    for (uint64_t w = 0; w < words_count; ++w)
    {
      std::getline(ifs, buf, ' '); // читаем слово (до пробела)
      vocab.push_back(buf);
      // читаем вектор
      float* eOffset = embeddings + w*emb_size;
      if ( fmt != "txt" )
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
  // функциясохранения и экспорта
  void save( const std::string& model_fn, const std::string& fmt = "c2v" )
  {
    FILE *fo = fopen(model_fn.c_str(), "wb");
    if (fmt == "c2v")
      fprintf(fo, "%lu %lu %lu %lu %lu\n", words_count-do_not_save.size(), emb_size, dep_size, assoc_size, gramm_size);
    else
      fprintf(fo, "%lu %lu\n", words_count-do_not_save.size(), emb_size);
    for (size_t a = 0; a < vocab.size(); ++a)
      if ( do_not_save.find(a) == do_not_save.end() )
        VectorsModel::write_embedding(fo, vocab[a], &embeddings[a * emb_size], emb_size, (fmt == "txt"));
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
  // поиск слова в словаре (в неизменяемой! модели с построением индекса)
  size_t get_word_idx_fast(const std::string& word) const
  {
    static std::map<std::string, size_t> vmap;
    if (words_count == 0) return 0;
    if (vmap.size() == 0)
    {
      for (size_t idx = 0; idx < words_count; ++idx)
        vmap[ vocab[idx] ] = idx;
    }
    auto it = vmap.find(word);
    if (it == vmap.end()) return words_count;
    return it->second;
  } // method-end
  // слияние моделей
  bool merge(const VectorsModel& ext)
  {
    if (emb_size != ext.emb_size)
    {
      std::cerr << "Models merging with different embedding sizes" << std::endl;
      return false;
    }
    size_t new_data_future_offset = words_count * emb_size;
    // при наличии дубликатов: вектора из новой (вливаемой) модели замещают старые (текущие)
    // используется, в частности, для безопасного добавления знаков препинания
    for (auto w : ext.vocab)
    {
      size_t vec_idx = get_word_idx(w);
      if (vec_idx != vocab.size())
        do_not_save.insert(vec_idx);
    }
    // копируем словарь
    std::copy(ext.vocab.begin(), ext.vocab.end(), std::back_inserter(vocab));
    words_count = vocab.size();
    // перевыделяем память и копируем новые эмбеддинги
    embeddings = (float *) realloc(embeddings, words_count * emb_size * sizeof(float) );
    if (embeddings == nullptr)
    {
      report_alloc_error();
      return false;
    }
    std::copy(ext.embeddings, ext.embeddings + ext.words_count*ext.emb_size, embeddings+new_data_future_offset);
    return true;
  }
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
  static void write_embedding( FILE* fo, const std::string& word, float* embedding, size_t emb_size, bool useTxtFmt = false )
  {
    write_embedding_slice( fo, word, embedding, 0, emb_size, useTxtFmt );
  } // method-end
  static void write_embedding_slice( FILE* fo, const std::string& word, float* embedding, size_t begin, size_t end, bool useTxtFmt = false )
  {
    write_embedding__start(fo, word, useTxtFmt);
    write_embedding__vec(fo, embedding, begin, end, useTxtFmt);
    write_embedding__fin(fo);
  } // method-end
  static void write_embedding__start( FILE* fo, const std::string& word, bool useTxtFmt = false )
  {
    fprintf(fo, "%s", word.c_str());
    if ( !useTxtFmt )
      fprintf(fo, " ");
  } // method-end
  static void write_embedding__vec( FILE* fo, float* embedding, size_t begin, size_t end, bool useTxtFmt = false )
  {
    for (size_t b = begin; b < end; ++b)
    {
      if ( useTxtFmt )
        fprintf(fo, " %lf", embedding[b]);
      else
        fwrite(&embedding[b], sizeof(float), 1, fo);
    }
  } // method-end
  static void write_embedding__fin( FILE* fo )
  {
    fprintf(fo, "\n");
  } // method-end
private:
  void report_alloc_error() const
  {
    std::cerr << "Cannot allocate memory: " << (words_count * emb_size * sizeof(float) / 1048576) << " MB" << std::endl;
    std::cerr << "    Words: " << words_count << std::endl;
    std::cerr << "    Embedding size: " << emb_size << std::endl;
  }
}; // class-decl-end


#endif /* VECTORS_MODEL_H_ */
