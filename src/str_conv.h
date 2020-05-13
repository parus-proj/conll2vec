#ifndef STR_CONV_H_
#define STR_CONV_H_

#include <string>
#include <bits/locale_conv.h>        // определение std::wstring_convert (todo: странность gcc?)
#include <codecvt>

class StrConv
{
public:
  // конвертеры кодировок UTF-8  <--> UTF-32
  static std::string To_UTF8(const std::u32string &s)
  {
      std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
      return conv.to_bytes(s);
  }
  static std::u32string To_UTF32(const std::string &s)
  {
      std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
      return conv.from_bytes(s);
  }
};


#endif /* STR_CONV_H_ */
