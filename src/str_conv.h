#ifndef STR_CONV_H_
#define STR_CONV_H_

#include <string>
#ifdef _MSC_VER
  #include <codecvt>
#else
  #include <bits/locale_conv.h>        // определение std::wstring_convert (todo: странность gcc?)
  #include <codecvt>
#endif

class StrConv
{
public:
  // конвертеры кодировок UTF-8  <--> UTF-32
  static std::string To_UTF8(const std::u32string &s)
  {
      #if _MSC_VER >= 1900 && _MSC_VER < 2000
        static std::wstring_convert<std::codecvt_utf8<__int32>, __int32> conv;
        auto p = reinterpret_cast<const int32_t *>(s.data());
        return conv.to_bytes(p, p + s.size());
      #else
        static std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
        return conv.to_bytes(s);
      #endif
  }
  static std::u32string To_UTF32(const std::string &s)
  {
      #if _MSC_VER >= 1900 && _MSC_VER < 2000
        static std::wstring_convert<std::codecvt_utf8<__int32>, __int32> conv;
        auto r = conv.from_bytes(s);
        return reinterpret_cast<const char32_t *>(r.data());
      #else
        static std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
        return conv.from_bytes(s);
      #endif
  }
};


#endif /* STR_CONV_H_ */
