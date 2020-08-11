#ifndef ADD_PUNCT_H_
#define ADD_PUNCT_H_

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
    std::vector<std::string> vocab;
    float *embeddings;
    size_t words_count;
    size_t emb_size;
    // открываем файл модели
    std::ifstream ifs(model_fn.c_str(), std::ios::binary);
    if ( !ifs.good() )
    {
      std::cerr << "Model file not found" << std::endl;
      return;
    }
    std::string buf;
    // считыавем размер матрицы
    ifs >> words_count;
    ifs >> emb_size;
    std::getline(ifs,buf); // считываем конец строки
    // выделяем память для эмбеддингов
    embeddings = (float *) malloc( words_count * emb_size * sizeof(float) );
    if (embeddings == NULL)
    {
      std::cerr << "Cannot allocate memory: " << (words_count * emb_size * sizeof(float) / 1048576) << " MB" << std::endl;
      std::cerr << "    Words: " << words_count << std::endl;
      std::cerr << "    Embedding size: " << emb_size << std::endl;
      return;
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
    }
    ifs.close();

    // 2. Добавляем в модель знаки пунктуации
    const size_t puncts_count = 2;
    std::vector<std::string> new_vocab;
    float *new_embeddings = (float *) malloc( puncts_count * emb_size * sizeof(float) );

    add_as_neighbour(vocab, ".", "во-вторых", embeddings, emb_size, new_vocab, new_embeddings);
    add_as_neighbour(vocab, ",", "и", embeddings, emb_size, new_vocab, new_embeddings);

    // 3. Сохраняем модель, расширенную знаками пунктуации
    size_t old_vocab_size = std::count_if(vocab.begin(), vocab.end(), [](const std::string& item) {return !item.empty();});
    FILE *fo = fopen(model_fn.c_str(), "wb");
    fprintf(fo, "%lu %lu\n", old_vocab_size+new_vocab.size(), emb_size);
    for (size_t a = 0; a < vocab.size(); ++a)
    {
      if ( vocab[a].empty() )
        continue;
      fprintf(fo, "%s ", vocab[a].c_str());
      for (size_t b = 0; b < emb_size; ++b)
      {
        if ( !useTxtFmt )
          fwrite(&embeddings[a * emb_size + b], sizeof(float), 1, fo);
        else
          fprintf(fo, " %lf", embeddings[a * emb_size + b]);
      }
      fprintf(fo, "\n");
    }
    for (size_t a = 0; a < new_vocab.size(); ++a)
    {
      fprintf(fo, "%s ", new_vocab[a].c_str());
      for (size_t b = 0; b < emb_size; ++b)
      {
        if ( !useTxtFmt )
          fwrite(&new_embeddings[a * emb_size + b], sizeof(float), 1, fo);
        else
          fprintf(fo, " %lf", new_embeddings[a * emb_size + b]);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);

  } // method-end

private:
  static size_t get_word_idx(const std::vector<std::string>& vocab, const std::string& word)
  {
    size_t widx = 0;
    for ( ; widx < vocab.size(); ++widx )
      if (vocab[widx] == word)
        break;
    return widx;
  } // method-end

  static void add_as_neighbour( std::vector<std::string>& vocab, const std::string& punct, const std::string& mount_point,
                                float* embeddings, size_t emb_size,
                                std::vector<std::string>& new_vocab, float* new_embeddings )
  {
    // проверяем наличие вектора для знака препинания в модели (если есть, то затрём его)
    size_t vec_idx = get_word_idx(vocab, punct);
    if (vec_idx != vocab.size())
      vocab[vec_idx].clear();
    // находим опорную точку, рядом с которой разместим знак препинания
    size_t mnt_idx = get_word_idx(vocab, mount_point);
    if (vec_idx == vocab.size())
    {
      std::cerr << "Mount-point not found:   punct=" << punct << "   mnt=" << mount_point << std::endl;
      return;
    }
    // порождаем словарную запись и вектор для знака препинания
    float *mnt_offset = embeddings + mnt_idx * emb_size;
    float *new_offset = new_embeddings + new_vocab.size() * emb_size;
    auto random_sign = []() -> float { return ((float)(rand() % 2) - 0.5)/0.5; };
    for (size_t d = 0; d < emb_size; ++d)
    {
      float *mnt_dim = mnt_offset + d;
      *(new_offset + d) = *mnt_dim + random_sign() * (*mnt_dim / 100);
    }
    new_vocab.push_back(punct);
  } // method-end

}; // class-decl-end


#endif /* ADD_PUNCT_H_ */
