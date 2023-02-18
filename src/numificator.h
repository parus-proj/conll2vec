#ifndef NUMIFICATOR_H_
#define NUMIFICATOR_H_

#include <string>
#include <cstring>       // for std::strerror

class Numificator
{
public:
  // обобщение токенов, содержащих числовые величины
  // превращаем слова вида 15-летие в @num@-летие
  static std::u32string process(const std::u32string& str)
  {
    const std::u32string NUM  = U"@num@";
    const std::u32string Head = U"0123456789‐‒–—-−˗―─,";
    const std::u32string Tail = U"АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯабвгдеёжзийклмнопрстуфхцчшщьыъэюя-‐";
    // быстрая проверка на наличие дефисоида
    size_t hyphenPos = str.find_first_of(U"‐‒–—-−˗―─"); // здесь всё: hyphens, dashes, minuses, boxlines
    if (hyphenPos == std::u32string::npos)
      return str;
    // верифицируем структуру
    //   спереди могут быть цифры и разделители (запятая, дефисоиды)
    //   сзади -- буквы и дефисы
    //   перед и зад должны сходиться на единственном общем дефисоиде
    size_t frwdPos = str.find_first_not_of(Head);
    if ( frwdPos == std::string::npos ) return str;
    size_t bkwdPos = str.find_last_not_of(Tail);
    if ( bkwdPos == std::string::npos ) return str;
    if ( frwdPos-1 == bkwdPos+1 )
        return U"@num@-" + str.substr(frwdPos);
    else
      return str;
  } // method-end
}; // class-decl-end


#endif /* NUMIFICATOR_H_ */
