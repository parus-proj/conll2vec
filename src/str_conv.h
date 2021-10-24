#ifndef STR_CONV_H_
#define STR_CONV_H_

#include <string>
#include <locale>
#include <codecvt>
#include <unicode/uchar.h>


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
  // конвертация строки в нижний регистр
  static std::u32string toLower(const std::u32string& str)
  {
    size_t len = str.length();
    std::u32string result = str;
    for (size_t idx = 0; idx < len; ++idx)
    {
      if (u_isUUppercase(result[idx]))
        result[idx] = u_tolower(result[idx]);  //TODO: лучше использовать u_strToLower (т.к. она учитывает соседние литеры), но это еще одна конверсия строки... подумать
    }
    return result;
  }
  // левый trim (in place)
  static inline void ltrim(std::string &s)
  {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
  }
  // правй trim (in place)
  static inline void rtrim(std::string &s)
  {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
  }
  // trim (in place)
  static inline void trim(std::string &s)
  {
    ltrim(s);
    rtrim(s);
  }
};


#endif /* STR_CONV_H_ */
