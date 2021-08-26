#ifndef SSEVAL_H_
#define SSEVAL_H_

#include "original_word2vec_vocabulary.h"

#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>

// Вывод данных о subsampling применительно к данному словарю
class SsEval
{
public:
  static void run( const std::string& vocab_fn )
  {
    OriginalWord2VecVocabulary vocab;
    if ( !vocab.load(vocab_fn) )
      return;
    if ( vocab.size() == 0 )
      return;

    auto thr_func = [vocab](size_t thr)
        {
          for (size_t i = 0; i < vocab.size(); ++i)
            if ( vocab.idx_to_data(i).cn < thr )
              return i*100.0/vocab.size();
          return 100.0;
        };

    std::cout << std::endl;
    std::cout << "1 000 000 threshold = " << thr_func(1000000) << " %" << std::endl;
    std::cout << "100 000   threshold = " << thr_func(100000) << " %" << std::endl;
    std::cout << "10 000    threshold = " << thr_func(10000) << " %" << std::endl;
    std::cout << "1 000     threshold = " << thr_func(1000) << " %" << std::endl;

    std::cout << std::endl;
    std::cout << "Frequency sum = " << vocab.cn_sum() << std::endl;

    std::cout << std::endl;
    std::vector<size_t> chp = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    std::cout << "Max frequency = " << vocab.idx_to_data(0).cn << std::endl;
    for (auto i : chp)
      std::cout << i << "% frequency = " << vocab.idx_to_data(vocab.size()*i/100).cn << std::endl;
    std::cout << "Min frequency = " << vocab.idx_to_data(vocab.size()-1).cn << std::endl;

    std::cout << std::endl;
    std::cout << "Probabilities table" << std::endl;
    std::cout << "subsamping\tmax";
    for (auto i : chp) std::cout << "\t" << i << "%";
    std::cout << "\tmin" << std::endl;
    std::vector<float> ss = {0, 1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 1e-7, 1e-8, 1e-9};
    for (auto s : ss)
    {
      vocab.sampling_estimation(s);
      std::cout << std::scientific << std::setprecision(0) << s << "\t\t" << std::fixed << std::setprecision(4) << vocab.idx_to_data(0).sample_probability;
      for (auto i : chp)
        std::cout << "\t" << vocab.idx_to_data(vocab.size()*i/100).sample_probability;
      std::cout << "\t" << vocab.idx_to_data(vocab.size()-1).sample_probability;
      std::cout << std::endl;
    }

  } // method-end
}; // class-decl-end


#endif /* SSEVAL_H_ */
