#ifndef MWE_VOCABULARY_H_
#define MWE_VOCABULARY_H_

#include "str_conv.h"
#include "original_word2vec_vocabulary.h"
#include "learning_example.h"

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <iostream>
#include <optional>


// представление узла синтаксического дерева
class TreeNode
{
public:
  std::weak_ptr<TreeNode> parent_tmp;                 // временное поле, используемое при построении дерева
  std::vector< std::shared_ptr<TreeNode> > children;  // синтаксические потомки узла
  std::shared_ptr<TreeNode> head;                     // синтаксический предок узла
  bool out_of_match;                                  // если true, то узел служит только для целей распознавания словосочетания и не подлежит замещению
  bool tok_match;                                     // сопоставлять по полю токена (не леммы)
  std::string word;
  TreeNode(const std::string& word_str, std::shared_ptr<TreeNode> parent_ptr)
  : parent_tmp(parent_ptr)
  , out_of_match(true)
  , tok_match(false)
  , word(word_str)
  {
  }
};

// представление лексикализованного словосочетания или значения маркированной вершины
class Phrase
{
public:
  // дескриптор
  std::string str;
  // список узлов, соответствующих маркированным вершинам синтаксических деревьев
  std::vector< std::shared_ptr<TreeNode> > trees;
  // лемма маркированной вершины
  std::string main_lemma;
  Phrase()
  {
  }
  // отладочная печать
  void dbg_print() const
  {
    std::cout << str << "  --";
    for (auto t : trees)
    {
      std::cout << "  ";
      printNode(t);
    }
  }
  void printNode(const std::shared_ptr<TreeNode> node) const
  {
    std::cout << (node->out_of_match ? "[" : "{");
    if (node->head)
    {
      std::cout << "^";
      printNode(node->head);
    }
    std::cout << node->word;
    for (auto ch : node->children)
      printNode(ch);
    std::cout << (node->out_of_match ? "]" : "}");
  }
};


// todo: для начала сделаем примитив на нормальных формах только; затем более тщательный поиск словосочетания

class MweVocabulary
{
public:
  // c-tor
  MweVocabulary( )
  {
  }
  // загрузка словаря
  bool load(const std::string& fn, std::shared_ptr< OriginalWord2VecVocabulary > main_vocabulary = nullptr)
  {
    std::ifstream wme_ifs(fn.c_str());
    std::string line;
    std::optional< std::multimap<std::string, std::shared_ptr<Phrase>>::iterator> last_inserted_mwe;
    while ( std::getline(wme_ifs, line).good() )
    {
      // отрежем комментарий
      size_t comment_sign_pos = line.find('#');
      if (comment_sign_pos != std::string::npos)
        line.erase( comment_sign_pos );
      // подчистим строку, после отрезания комментария (перед ним могут быть пробелы)
      StrConv::trim(line);
      if (line.empty()) continue;

      // разбиваем запись по символу табуляции
      size_t tab1 = line.find_first_of("\t");
      if (tab1 == std::string::npos) // пропускаем некорректные записи
      {
        last_inserted_mwe.reset();
        std::cerr << "mwe: invalid line: " << line << std::endl;
        continue;
      }

      std::string descr = line.substr(0, tab1);
      std::shared_ptr<TreeNode> tree = str2tree( line.substr(tab1+1) );

      if ( descr != "+" )
      {
        std::shared_ptr<Phrase> phrase = std::make_shared<Phrase>();
        phrase->str = descr;
        phrase->trees.push_back(tree);
        phrase->main_lemma = tree->word;
        last_inserted_mwe = mwes.insert( std::make_pair(tree->word, phrase) );
      }
      else
      {
        if ( !last_inserted_mwe ) // пропускаем некорректные записи
        {
          std::cerr << "mwe: invalid line: " << line << std::endl;
          continue;
        }
        auto& phrase = (*last_inserted_mwe)->second;
        if ( tree->word != phrase->main_lemma ) // контроль того, что вершина та же
        {
          std::cerr << "mwe: head change error: " << line << std::endl;
          continue;
        }
        phrase->trees.push_back(tree);
      }
    }

    // профильтруем фразы, чтобы удовлетворяли частотному порогу главного словаря (если он уже построен)
    // в этом случае put_phrases_into_sentence не будет преобразовывать лемму/предложение, если фраза не попала в главный словарь
    if (main_vocabulary)
    {
      const size_t INVALID_IDX = std::numeric_limits<size_t>::max();
      auto it = mwes.begin();
      while (it != mwes.end())    //TODO: в C++20 заменить на std::erase_if (https://en.cppreference.com/w/cpp/container/map/erase_if)
      {
        auto& descr = it->second->str;
        if ( main_vocabulary->word_to_idx(descr) != INVALID_IDX )
          ++it;
        else
          it = mwes.erase(it);
      }
    }

    return true;
  } // method-end
  // поиск фраз в предложении и встраивание их туда
  void put_phrases_into_sentence( std::vector< std::vector<std::string> >& sentence_matrix ) const
  {
    // Переделываем само предложение следующим образом.
    // 1) Там, где обнаруживается словосочетание, являющееся лексической единицей, оно полностью вытесняется из предложения и замещается
    // одним токеном (необходимые синтаксические ссылки исправляются).
    // 2) Если словосочетание диагностическое, то вместо леммы слова, значение которого уточняется, подставляется дескриптор словосочетания.
    // Дескриптор фактически заменяет одно из значений многозначного слова.

    // сначала ищем каждое слово предложения в индексе маркир.вершин словосочетаний
    // формируем short-list словосочетаний, которые нужно поискать в предложении
    std::map< size_t, std::vector<std::shared_ptr<Phrase>> > phCandidates;  // отображение из индекса токена предложения в список фраз-кандидатов
    ph2s_search_candidates(sentence_matrix, phCandidates);
    if ( phCandidates.empty() )
      return;

    // строим для рассматриваемого предложения мэппинг из токена в список его синтаксических потомков
    // (это вспомогательная структура для ускорения поиска синтакс. потомков заданного слова)
    std::map< size_t, std::vector<size_t> > deps;
    if ( !ph2s_build_deps(sentence_matrix, deps) )
      return;

    // пытаемся найти фразы-кандидаты в синтакс.дереве предложения
    std::set<size_t> match;
    while ( !phCandidates.empty() )
    {
      auto c = *phCandidates.begin();
      phCandidates.erase(phCandidates.begin());
      for (auto& ph : c.second)
      {
        if ( compare_trees(sentence_matrix, deps, c.first, ph, match) )
        {
//          dbg_print_sentence(sentence_matrix);
//          dbg_print_sentence_conll(sentence_matrix);
          if ( !match.empty() )
          {
            ph2s_replace(sentence_matrix, c.first, match, ph->str);
            // если струкутура предложения поменялась, индексы во вспомогательных структурах (phCandidates, deps) могут стать невалидными
            // перестроим их по уже скорректированной sentence_matrix
            ph2s_search_candidates(sentence_matrix, phCandidates);
            if ( phCandidates.empty() ) return;
            if ( !ph2s_build_deps(sentence_matrix, deps) ) return;
            break;
          }
          else
            sentence_matrix[c.first][2] = ph->str;
//          dbg_print_sentence_conll(sentence_matrix);
//          static size_t dbg_cnt = 0;
//          if (++dbg_cnt == 10)
//            exit(0);
        }
      }
    }

  } // method-end
  // вспомогательный метод для incorporate_phrases_to_sentence
  // строит отображение из индекса токена предложения в список фраз-кандидатов
  void ph2s_search_candidates(const std::vector< std::vector<std::string> >& sentence_matrix, std::map< size_t, std::vector< std::shared_ptr<Phrase> > >& phCandidates) const
  {
    phCandidates.clear();
    for (size_t tidx = 0; tidx < sentence_matrix.size(); ++tidx)
    {
      auto& norma = sentence_matrix[tidx][2];
      auto range = mwes.equal_range(norma);
      for (auto i = range.first; i != range.second; ++i)
        phCandidates[tidx].push_back(i->second);
    }
  } // method-end
  // вспомогательный метод для incorporate_phrases_to_sentence
  // строит структуру для быстрого поиска зависимых данной вершины дерева
  bool ph2s_build_deps(const std::vector< std::vector<std::string> >& sentence_matrix, std::map< size_t, std::vector<size_t> >& deps) const
  {
    deps.clear();
    try
    {
      for (size_t tidx = 0; tidx < sentence_matrix.size(); ++tidx)
        deps[ std::stoi(sentence_matrix[tidx][6]) - 1 ].push_back(tidx);
    } catch (...) {
      return false;
    }
    return true;
  } // method-end
  // вспомогательный метод для incorporate_phrases_to_sentence
  // замещает словосочетание в предложении
  void ph2s_replace( std::vector< std::vector<std::string> >& sentence_matrix,
                     size_t marked_position,
                     std::set<size_t> match,
                     const std::string& descr) const
  {
    // строим структуру для хрениния номеров токенов в поле синтаксических ссылок
    std::vector<size_t> heads;
    for (size_t idx = 0; idx < sentence_matrix.size(); ++idx)
      heads.push_back( std::stoi(sentence_matrix[idx][6]) );
    // всякую синтаксическую связь, ведущую в match, перекидываем на токен-дескриптор
    for (auto& h : heads)
      if ( match.find(h-1) != match.end() )
        h = marked_position + 1;
    // выкидываем из sentence_matrix все токены из match, кроме marked_position
    for (auto it = match.rbegin(); it != match.rend(); ++it)
    {
      size_t pos = *it;
      if (pos == marked_position)
      {
        sentence_matrix[marked_position][2] = descr;
        continue;
      }
      sentence_matrix.erase(sentence_matrix.begin() + pos);
      heads.erase(heads.begin() + pos);
      // перенумеруем токены, после удаления
      for (size_t idx = pos; idx < sentence_matrix.size(); ++idx)
        sentence_matrix[idx][0] = std::to_string(idx+1);
      for (auto& h : heads)
        if (h > pos+1)
          --h;
    }
    // переносим откорректированные синтаксические ссылки в sentence_matrix
    for (size_t idx = 0; idx < heads.size(); ++idx)
      sentence_matrix[idx][6] = std::to_string(heads[idx]);
  } // method-end

  // вычисление групп векторов, для которых необходимо выполнить свёртывание
  // (свёртывание временно выделенных словосочетаний к единому вектору вершины)
  void process_transient(std::shared_ptr< OriginalWord2VecVocabulary > main_vocabulary, std::vector< std::vector< std::pair<size_t, float> > >& collapsing_info)
  {
    const size_t OUT_OF_VOCABULARY = std::numeric_limits<size_t>::max();
    collapsing_info.clear();
    for (auto it = mwes.begin(); it != mwes.end(); )
    {
      // выделяем блок словосочетаний с общей вершиной, не вошедших в словарь дистрибутивной модели (т.наз., временных словосочетаний)
      auto range = mwes.equal_range(it->first);
      it = range.second;
      std::vector<std::shared_ptr<Phrase>> transients;
      for (auto i = range.first; i != range.second; ++i)
      {
        // будем ориентироваться по первому дереву (варианту) фразы
        auto first_tree = *(i->second->trees.begin());
        if ( first_tree->out_of_match ) // если узел верхнего уровня не вошел в словарь модели (т.е. временный)
          transients.push_back(i->second);
      }
      if ( transients.empty() ) // если только словарные словосочетания с такой вершиной
        continue;
      // находим веса вершины и временных словосочетаний
      // (для последующего вычисления взвешенного среднего между вектором вершины и векторами временных словосочетаний)
std::cout << std::endl << "COLLAPSING: " << transients.front()->main_lemma << std::endl;
      size_t head_idx = main_vocabulary->word_to_idx( transients.front()->main_lemma );
      if ( head_idx == OUT_OF_VOCABULARY ) continue;
      uint64_t head_sum = main_vocabulary->idx_to_data( head_idx ).cn;  // вычислим количество упоминаний вершины вне словосочетаний (без учёта сабсэмплинга, т.к. meet_counter также вычисляются без учёта сабсэмплинга)
      size_t total_sum = head_sum;
      for (auto& t : transients)
      {
        size_t phrase_idx = main_vocabulary->word_to_idx( t->str );
        if ( phrase_idx == OUT_OF_VOCABULARY ) continue;
        total_sum += main_vocabulary->idx_to_data( phrase_idx ).cn;
      }
      std::vector< std::pair<size_t, float> > vectors_indexes_and_weights;
      vectors_indexes_and_weights.push_back( std::make_pair( head_idx, (float)head_sum/(float)total_sum ) );
std::cout << "  " << transients.front()->main_lemma << ", " << ((float)head_sum/(float)total_sum) << std::endl;
      for (auto& t : transients)
      {
        size_t phrase_idx = main_vocabulary->word_to_idx( t->str );
        if ( phrase_idx == OUT_OF_VOCABULARY ) continue;
        size_t phrase_cn = main_vocabulary->idx_to_data( phrase_idx ).cn;
        vectors_indexes_and_weights.push_back( std::make_pair( phrase_idx, (float)phrase_cn/(float)total_sum ) );
std::cout << "  " << t->str << ", " << ((float)phrase_cn/(float)total_sum) << std::endl;
      }
      collapsing_info.push_back(vectors_indexes_and_weights);
    }
  } // method-end


  void dbg_print_mwe_lists()
  {
    std::cout << std::endl;
    for (auto& mwe : mwes)
    {
      mwe.second->dbg_print();
      std::cout << std::endl;
    }
  }

private:
  // хранилище словосочетаний в древесной форме (проидексированных по вершинам словосочетаний)
  std::multimap<std::string, std::shared_ptr<Phrase>> mwes;

  // функция построения дерева по его строковому представлению
  std::shared_ptr<TreeNode> str2tree(const std::string& str) const
  {
    std::shared_ptr<TreeNode> holder = std::make_shared<TreeNode>("", nullptr);
    std::shared_ptr<TreeNode> currNode = holder;
    size_t pos = 0;
    bool inside_node_constraint = false;
    auto nextTokenFunc = [](const std::string& str, size_t& pos) -> std::string
                         {
                           const std::set<char> STRUCT_TOKENS = {'^','[',']','{','}', '(', ')'};
                           if ( str[pos] == '^' )
                           {
                             ++pos;
                             if (pos == str.length() || (str[pos] != '[' && str[pos] != '{'))
                             {
                               std::cerr << "Fatal error: invalid mwe record:" << str << std::endl;
                               exit(-1);
                             }
                             ++pos;
                             return str.substr(pos-2, 2);
                           }
                           else if ( str[pos] == '[' ) { ++pos; return "["; }
                           else if ( str[pos] == ']' ) { ++pos; return "]"; }
                           else if ( str[pos] == '{' ) { ++pos; return "{"; }
                           else if ( str[pos] == '}' ) { ++pos; return "}"; }
                           else if ( str[pos] == '(' ) { ++pos; return "("; }
                           else if ( str[pos] == ')' ) { ++pos; return ")"; }
                           else {
                             std::string result;
                             while ( STRUCT_TOKENS.find(str[pos]) == STRUCT_TOKENS.end() )
                             {
                               result += str[pos++];
                               if (pos == str.length())
                               {
                                 std::cerr << "Fatal error: invalid mwe record:" << str << std::endl;
                                 exit(-1);
                               }
                             }
                             return result;
                           }
                         };
    while ( pos != str.length() )
    {
      std::string tok = nextTokenFunc(str, pos);
      if ( tok == "[" || tok == "{" )
      {
        std::shared_ptr<TreeNode> newNode = std::make_shared<TreeNode>("", currNode);
        newNode->out_of_match = (tok == "[");
        currNode->children.push_back( newNode );
        currNode = newNode;
      }
      else if ( tok == "^[" || tok == "^{" )
      {
        std::shared_ptr<TreeNode> newNode = std::make_shared<TreeNode>("", currNode);
        newNode->out_of_match = (tok == "^[");
        currNode->head = newNode;
        currNode = newNode;
      }
      else if ( tok == "]" || tok == "}" )
      {
        if ( currNode->word.empty() )
          std::cerr << "mwe: empty node error: " << str << std::endl;
        currNode = currNode->parent_tmp.lock();
      }
      else if ( tok == "(" )
        inside_node_constraint = true;
      else if ( tok == ")" )
        inside_node_constraint = false;
      else
      {
        if (inside_node_constraint)
        {
          currNode->tok_match = tok.length() > 0 && tok[0] == 't';
        }
        else
        {
          if ( !currNode->word.empty() )
            std::cerr << "mwe: hierarchy error: " << str << std::endl;
          currNode->word = tok;
        }
      }
    } // while str isn't finished
    return holder->children[0];
  } // method-end
  // проверка вхождения словосочетания в заданную позицию предложения
  // при нахождении словосочетания возвращает множество индексов токенов, составляющих часть фразы, подлежащей замене на дескриптор
  bool compare_trees( const std::vector< std::vector<std::string> >& sentence_matrix,
                      const std::map< size_t, std::vector<size_t> >& deps,
                      size_t match_point,
                      std::shared_ptr<Phrase> phrase,
                      std::set<size_t>& match_result ) const
  {
    for (auto& t : phrase->trees)
    {
      bool succ = compare_trees_helper(sentence_matrix, deps, match_point, t, match_result);
      if (succ) return true;
    }
    return false;
  } // method-end
  bool compare_trees_helper( const std::vector< std::vector<std::string> >& sentence_matrix,
                             const std::map< size_t, std::vector<size_t> >& deps,
                             size_t match_point,
                             std::shared_ptr<TreeNode> tree,
                             std::set<size_t>& match_result ) const
  {
    // исходным состоянием является соответствие между match_point и вершиной tree (они уже сопоставлены)

    match_result.clear();
    if ( !tree->out_of_match )
      match_result.insert(match_point);

    // список подлежащих поиску узлов в виде кортежа <уже сопоставленный индекс токена, искомый узел, направление поиска>
    std::queue< std::tuple<size_t, std::shared_ptr<TreeNode>, bool> > need_to_match;
    auto add_match_query = [&need_to_match](size_t the_pos, std::shared_ptr<TreeNode> the_node)
                           {
                             if (the_node->head)
                               need_to_match.push( std::make_tuple(the_pos, the_node->head, true));
                             for (auto& n : the_node->children)
                               need_to_match.push( std::make_tuple(the_pos, n, false) );
                           };

    add_match_query(match_point, tree);

    // используем жадный алгоритм сопоставления
    // допустим ищется словосочетание [пригласить[друг]]
    // во фразе "друг пригласил друга на чай" у слова "пригласить" два потомка "друг"
    // алгоритм попытается сопоставиться с первым попавшимся "друг", и если не сопоставит, то альтернативный вариант сопоставления рассматриваться не будет
    // предполагается, что множественные сопоставления редки
    // WARNING: возможно конструкции с предлогами будут проблемными: пригласил в четверг в бар

    while ( !need_to_match.empty() )
    {
      size_t actual_token_no = std::get<0>(need_to_match.front());
      std::shared_ptr<TreeNode> tree_node = std::get<1>(need_to_match.front());
      bool search_up = std::get<2>(need_to_match.front());
      size_t text_field_idx = tree_node->tok_match ? 1 : 2;
      need_to_match.pop();
      if (search_up)
      {
        int syn_head = std::stoi(sentence_matrix[actual_token_no][6]) - 1;
        if (syn_head < 0 || sentence_matrix[syn_head][text_field_idx] != tree_node->word)
        {
          match_result.clear();
          return false;
        }
        add_match_query(syn_head, tree_node);
        if ( !tree_node->out_of_match )
          match_result.insert(syn_head);
      }
      else
      {
        auto deps_it = deps.find(actual_token_no);
        if ( deps_it == deps.end() ) { match_result.clear(); return false; } // у того, кто должен быть родителем, нет потомков вообще
        bool found = false;
        for (auto d : deps_it->second)
          if (sentence_matrix[d][text_field_idx] == tree_node->word) // нашли зависимое
          {
            add_match_query(d, tree_node);
            if ( !tree_node->out_of_match )
              match_result.insert(d);
            found = true;
            break;
          }
        if (found)
          continue;
        match_result.clear();
        return false; // не нашли очередного зависимого
      }
    } // while match isn't complete
    return true;
  } // method-end


  // отладочные процедуры
  void dbg_print_sentence(const std::vector< std::vector<std::string> >& sentence_matrix) const
  {
    std::string txt;
    for (auto& t : sentence_matrix)
      txt += " " + t[1];
    txt.erase(0, 1);
    std::cout << txt << std::endl;
  }
  void dbg_print_sentence_conll(const std::vector< std::vector<std::string> >& sentence_matrix) const
  {
    for (auto& t : sentence_matrix)
    {
      for (auto& f : t)
        std::cout << f << "\t";
      std::cout << std::endl;
    }
  }
}; // class-decl-end



#endif /* MWE_VOCABULARY_H_ */
