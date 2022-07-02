#ifndef _Log_H_GUARD_
#define _Log_H_GUARD_

#include <string>
#include <fstream>
#include <mutex>

const bool ALLOW_LOG = true;

#define NONCOPYABLE(Type) Type(const Type&)=delete; Type& operator=(const Type&)=delete

/** Класс "Протоколирование". */
class Log
{
  NONCOPYABLE(Log);
public:
  // функция доступа к синглетону
  static Log& getInstance()
  {
    static Log s;
    return s;
  }

  // функции записи
  Log& operator()( const std::string& record )
  {
    if( ALLOW_LOG )
    {
      std::lock_guard<std::mutex> guard(mut);
      (*ofs) << record << '\n';
      ofs->flush();
    }
    return *this;
  }
  Log& operator()( const char* record )
  {
    return operator()(std::string(record));
  }
  template<typename T>
  Log& operator()( T number )
  {
    return operator()( std::to_string( number ) );
  }

private:
  std::ofstream *ofs;
  std::mutex mut;
  Log()
  : ofs(nullptr)
  {
    if( ALLOW_LOG )
      ofs = new std::ofstream( "c2v.log" );
  }
  ~Log()
  {
    delete ofs; 
  }
};  // class-dec-end: Log


#endif

