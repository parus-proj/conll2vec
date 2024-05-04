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
        {"-task",         {"Values: fit, vocab, train, sim, ...", std::nullopt, std::nullopt}},
        {"-model",        {"The model <file>", std::nullopt, std::nullopt}},
        {"-train",        {"Training data <file>.conll", std::nullopt, std::nullopt}},
        {"-vocab_l",      {"Lemmas vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_t",      {"Tokens vocabulary <file>", std::nullopt, std::nullopt}},
        {"-tl_map",       {"Tokens-lemmas mapping <file>", "token2lemmas.map", std::nullopt}},
        {"-vocab_o",      {"OOV items vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocab_e",      {"Expressions vocabulary <file>", "./data/mwe.list", std::nullopt}},
        {"-vocab_d",      {"Dependency contexts vocabulary <file>", std::nullopt, std::nullopt}},
        {"-backup",       {"Save neural network weights to <file>", std::nullopt, std::nullopt}},
        {"-restore",      {"Restore neural network weights from <file>", std::nullopt, std::nullopt}},
        {"-min-count_l",  {"Min frequency in Lemmas vocabulary", "50", std::nullopt}},
        {"-min-count_t",  {"Min frequency in Tokens vocabulary", "50", std::nullopt}},
        {"-min-count_d",  {"Min frequency in Dependency vocabulary", "50", std::nullopt}},
        {"-min-count_o",  {"Min frequency in OOV vocabulary", "10000", std::nullopt}},
        {"-exclude_nums", {"Exclude digital numbers while fitting", "0", std::nullopt}},
        {"-max_oov_sfx",  {"Maximal suffix length in OOV vocabulary", "5", std::nullopt}},
        {"-col_ctx_d",    {"Dependency contexts vocabulary column (in conll)", "3", std::nullopt}},
        {"-use_deprel",   {"Include DEPREL field in dependency context", "1", std::nullopt}},
        {"-size_d",       {"Size of Categorial part of word vectors", "60", std::nullopt}},
        {"-size_a",       {"Size of Associative part of word vectors", "40", std::nullopt}},
        {"-size_g",       {"Size of Grammatical part of word vectors", "20", std::nullopt}},
        {"-negative_d",   {"Number of negative examples (dependency)", "5", std::nullopt}},
        {"-negative_a",   {"Number of negative examples (associative)", "5", std::nullopt}},
        {"-alpha",        {"Set the starting learning rate", "0.025", std::nullopt}},
        {"-iter",         {"Run more training iterations", "5", std::nullopt}},
        {"-sample_w",     {"Words subsampling threshold", "1e-4", std::nullopt}},
        {"-sample_d",     {"Dependency contexts subsampling threshold", "1e-4", std::nullopt}},
        {"-sample_a",     {"Associative contexts subsampling threshold", "1e-5", std::nullopt}},
        {"-threads",      {"Use <int> threads", "8", std::nullopt}},
        {"-fit_input",    {"<file>.conll to fit (or stdin)", std::nullopt, std::nullopt}},
        {"-a_ratio",      {"Associations contribution to similarity", "1.0", std::nullopt}},
        {"-g_ratio",      {"Grammatics contribution to similarity", "0.1", std::nullopt}},
        {"-st_yo",        {"Replace 'yo' in russe while self-testing", "0", std::nullopt}},
        {"-rfile",        {"Resulting file for import/export task", std::nullopt, std::nullopt}},
        {"-model_fmt",    {"word2vec model format (bin|txt) for import/export task", "bin", std::nullopt}},
        {"-sub_l",        {"Left range bound for sub-model", std::nullopt, std::nullopt}},
        {"-sub_r",        {"Right range bound for sub-model", std::nullopt, std::nullopt}},
        {"-fsim_file",    {"File with word pairs for fsim task", std::nullopt, std::nullopt}},
        {"-fsim_fmt",     {"File with word pairs format (detail|russe)", "detail", std::nullopt}},
        {"-eval_vocab",   {"A vocabulary for subsampling evaluation", std::nullopt, std::nullopt}},
        {"-rr_vocab",     {"Reliable rel-pairs vocabulary <file>", std::nullopt, std::nullopt}},
        {"-rr_min_sim",   {"Reliable rel-pairs minimal similarity", "0.6", std::nullopt}},
        {"-ca_vocab",     {"Safe lemmas vocabulary <file>", std::nullopt, std::nullopt}},
        {"-vocabs_tab",   {"External vocabs table <file>", "./data/vocabs.table", std::nullopt}}

    };
  }
};

#endif /* COMMAND_LINE_PARAMETERS_DEFS_H_ */
