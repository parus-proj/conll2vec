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
    const std::set<std::string> puncts = { ".", ",", "!", "?", ":", ";", "…", "...", "--", "—", "–", "‒",
                                           "'", "ʼ", "ˮ", "\"", "«", "»", "“", "”", "„", "‟", "‘", "’", "‚", "‛",
                                           "(", ")", "[", "]", "{", "}", "⟨", "⟩" };
    for (auto p : puncts)
    {
      // проверяем наличие вектора для знака препинания в модели (если есть, то затрём его)
      size_t vec_idx = get_word_idx(vocab, p);
      if (vec_idx != vocab.size())
        vocab[vec_idx].clear();
    }
    std::vector<std::string> new_vocab;
    float *new_embeddings = (float *) malloc( puncts.size() * emb_size * sizeof(float) );

    float *support_embedding = (float *) malloc(emb_size*sizeof(float));
    calc_support_embedding(words_count, emb_size, embeddings, support_embedding);
    float *dot_se = (float *) malloc(emb_size*sizeof(float));
    float *dash_se = (float *) malloc(emb_size*sizeof(float));
    float *quote_se = (float *) malloc(emb_size*sizeof(float));
    float *lquote_se = (float *) malloc(emb_size*sizeof(float));
    float *rquote_se = (float *) malloc(emb_size*sizeof(float));
    float *bracket_se = (float *) malloc(emb_size*sizeof(float));
    float *lbracket_se = (float *) malloc(emb_size*sizeof(float));
    float *rbracket_se = (float *) malloc(emb_size*sizeof(float));
    make_embedding_as_neighbour(emb_size, support_embedding, dot_se, 7);
    make_embedding_as_neighbour(emb_size, support_embedding, dash_se, 7);
    make_embedding_as_neighbour(emb_size, support_embedding, quote_se, 7);
    make_embedding_as_neighbour(emb_size, support_embedding, bracket_se, 7);
    make_embedding_as_neighbour(emb_size, quote_se, lquote_se, 3);
    make_embedding_as_neighbour(emb_size, quote_se, rquote_se, 3);
    make_embedding_as_neighbour(emb_size, bracket_se, lbracket_se, 3);
    make_embedding_as_neighbour(emb_size, bracket_se, rbracket_se, 3);

    make_embedding_as_neighbour(emb_size, dot_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back(".");
    make_embedding_as_neighbour(emb_size, dot_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("!");
    make_embedding_as_neighbour(emb_size, dot_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("?");
    make_embedding_as_neighbour(emb_size, dot_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back(";");
    make_embedding_as_neighbour(emb_size, dot_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("…");
    make_embedding_as_neighbour(emb_size, dot_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("...");
    make_embedding_as_neighbour(emb_size, dot_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back(",");
    make_embedding_as_neighbour(emb_size, dash_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back(":");
    make_embedding_as_neighbour(emb_size, dash_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("--");
    make_embedding_as_neighbour(emb_size, dash_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("—");
    make_embedding_as_neighbour(emb_size, dash_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("–");
    make_embedding_as_neighbour(emb_size, dash_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("‒");
    make_embedding_as_neighbour(emb_size, quote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("'");
    make_embedding_as_neighbour(emb_size, quote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("ʼ");
    make_embedding_as_neighbour(emb_size, quote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("ˮ");
    make_embedding_as_neighbour(emb_size, quote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("\"");
    make_embedding_as_neighbour(emb_size, lquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("«");
    make_embedding_as_neighbour(emb_size, rquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("»");
    make_embedding_as_neighbour(emb_size, lquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("“");
    make_embedding_as_neighbour(emb_size, rquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("”");
    make_embedding_as_neighbour(emb_size, lquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("„");
    make_embedding_as_neighbour(emb_size, rquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("‟");
    make_embedding_as_neighbour(emb_size, lquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("‘");
    make_embedding_as_neighbour(emb_size, rquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("’");
    make_embedding_as_neighbour(emb_size, lquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("‚");
    make_embedding_as_neighbour(emb_size, rquote_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("‛");
    make_embedding_as_neighbour(emb_size, lbracket_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("(");
    make_embedding_as_neighbour(emb_size, rbracket_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back(")");
    make_embedding_as_neighbour(emb_size, lbracket_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("[");
    make_embedding_as_neighbour(emb_size, rbracket_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("]");
    make_embedding_as_neighbour(emb_size, lbracket_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("{");
    make_embedding_as_neighbour(emb_size, rbracket_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("}");
    make_embedding_as_neighbour(emb_size, lbracket_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("⟨");
    make_embedding_as_neighbour(emb_size, rbracket_se, new_embeddings + emb_size * new_vocab.size()); new_vocab.push_back("⟩");


//    add_as_neighbour(vocab, ".", "во-вторых", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "!", "во-вторых", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "?", "во-вторых", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, ";", "во-вторых", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "…", "во-вторых", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "...", "во-вторых", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, ",", "и", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, ":", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "--", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "—", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "–", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "‒", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "'", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "ʼ", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "ˮ", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "\"", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "«", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "»", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "“", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "”", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "„", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "‟", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "‘", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "’", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "‚", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "‛", "цитировать", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "(", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, ")", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "[", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "]", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "{", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "}", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "⟨", "например", embeddings, emb_size, new_vocab, new_embeddings);
//    add_as_neighbour(vocab, "⟩", "например", embeddings, emb_size, new_vocab, new_embeddings);

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
    // находим опорную точку, рядом с которой разместим знак препинания
    size_t mnt_idx = get_word_idx(vocab, mount_point);
    if (mnt_idx == vocab.size())
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
      *(support_embedding + d) = rbound + 2.0;
    }
  } // method-end

  static void make_embedding_as_neighbour( size_t emb_size, float* base_embedding, float* new_embedding, float distance_factor = 1.0 )
  {
    auto random_sign = []() -> float { return ((float)(rand() % 2) - 0.5)/0.5; };
    for (size_t d = 0; d < emb_size; ++d)
    {
      float *offs = base_embedding + d;
      *(new_embedding + d) = *offs + random_sign() * (*offs / 100 * distance_factor);
    }
  } // method-end

}; // class-decl-end


#endif /* ADD_PUNCT_H_ */
