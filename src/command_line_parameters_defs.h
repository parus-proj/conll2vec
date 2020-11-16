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
        {"-vocab_m",      {"Lemmas main vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_p",      {"Lemmas proper names vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_t",      {"Tokens vocabulary <file>", std::nullopt, std::nullopt}},
        {"-tl_map",       {"Tokens to lemmas mapping <file>", "token_lemmas.map", std::nullopt}},
//        {"-vocab_e",      {"Expressions vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_d",      {"Dependency contexts vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_a",      {"Associative contexts vocabulary <file>", std::nullopt, std::nullopt}},
        {"-backup",       {"Save neural network weights to <file>", std::nullopt, std::nullopt}},
        {"-restore",      {"Restore neural network weights from <file>", std::nullopt, std::nullopt}},
        {"-min-count_m",  {"Min frequency in Lemmas main vocabulary", "50", std::nullopt}},
        {"-min-count_p",  {"Min frequency in Lemmas proper-names vocabulary", "50", std::nullopt}},
        {"-min-count_t",  {"Min frequency in Tokens vocabulary", "50", std::nullopt}},
        {"-min-count_d",  {"Min frequency in Dependency vocabulary", "50", std::nullopt}},
        {"-col_ctx_d",    {"Dependency contexts vocabulary column (in conll)", "3", std::nullopt}},
        {"-use_deprel",   {"Include DEPREL field in dependency context", "1", std::nullopt}},
        {"-size_d",       {"Size of Dependency part of word vectors", "75", std::nullopt}},
        {"-size_a",       {"Size of Associative part of word vectors", "25", std::nullopt}},
        {"-negative",     {"Number of negative examples", "5", std::nullopt}},
        {"-alpha",        {"Set the starting learning rate", "0.025", std::nullopt}},
        {"-iter",         {"Run more training iterations", "5", std::nullopt}},
        {"-sample_w",     {"Words subsampling threshold", "1e-3", std::nullopt}},
        {"-sample_d",     {"Dependency contexts subsampling threshold", "1e-3", std::nullopt}},
        {"-sample_a",     {"Associative contexts subsampling threshold", "1e-5", std::nullopt}},
        {"-threads",      {"Use <int> threads", "8", std::nullopt}},
        {"-fit_input",    {"<file>.conll to fit (or stdin)", std::nullopt, std::nullopt}},
        {"-a_ratio" ,     {"Associations contribution to similarity", "1.0", std::nullopt}},
        {"-st_yo" ,       {"Replace 'yo' in russe while self-testing", "0", std::nullopt}}
    };
  }
};

#endif /* COMMAND_LINE_PARAMETERS_DEFS_H_ */
