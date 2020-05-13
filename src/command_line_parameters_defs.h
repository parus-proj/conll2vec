#ifndef COMMAND_LINE_PARAMETERS_DEFS_H_
#define COMMAND_LINE_PARAMETERS_DEFS_H_

#include "command_line_parameters.h"

class CommandLineParametersDefs : public CommandLineParameters
{
public:
  CommandLineParametersDefs()
  {
    // initialize params mapping with std::initialzer_list<T>
    params_ = {
        {"-task",         {"Values: fit, vocab, train, punct, sim", std::nullopt, std::nullopt}},
        {"-model",        {"The model <file>", std::nullopt, std::nullopt}},
        {"-model_fmt",    {"The model format (bin|txt)", "bin", std::nullopt}},
        {"-train",        {"Training data <file>.conll", std::nullopt, std::nullopt}},
        {"-vocab_m",      {"Main vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_p",      {"Proper names vocabulary <file>", std::nullopt, std::nullopt}},
//        {"-vocab_e",      {"Expressions vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_d",      {"Dependency contexts vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_a",      {"Associative contexts vocabulary <file>", std::nullopt, std::nullopt}},
        {"-backup",       {"Save neural network weights to <file>", std::nullopt, std::nullopt}},
        {"-restore",      {"Restore neural network weights from <file>", std::nullopt, std::nullopt}},
        {"-min-count_m",  {"Min frequency in Main vocabulary", "50", std::nullopt}},
        {"-min-count_p",  {"Min frequency in Proper-names vocabulary", "50", std::nullopt}},
        {"-min-count_d",  {"Min frequency in Dependency vocabulary", "50", std::nullopt}},
        {"-min-count_a",  {"Min frequency in Associative vocabulary", "50", std::nullopt}},
        {"-col_emb",      {"Embeddings vocabulary column (in conll)", "2", std::nullopt}},
        {"-col_ctx_d",    {"Dependency contexts vocabulary column (in conll)", "3", std::nullopt}},
        {"-size_d",       {"Size of Dependency part of word vectors", "80", std::nullopt}},
        {"-size_a",       {"Size of Associative part of word vectors", "20", std::nullopt}},
        {"-negative",     {"Number of negative examples", "3", std::nullopt}},
        {"-alpha",        {"Set the starting learning rate", "0.025", std::nullopt}},
        {"-iter",         {"Run more training iterations", "3", std::nullopt}},
        {"-threads",      {"Use <int> threads", "8", std::nullopt}},
        {"-fit_result",   {"<file>.conll with fit results", std::nullopt, std::nullopt}}
    };
  }
};

#endif /* COMMAND_LINE_PARAMETERS_DEFS_H_ */
