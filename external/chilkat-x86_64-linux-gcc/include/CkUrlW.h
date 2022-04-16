// CkUrlW.h: interface for the CkUrlW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkUrlW_H
#define _CkUrlW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkUrlW
class CK_VISIBLE_PUBLIC CkUrlW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkUrlW(const CkUrlW &);
	CkUrlW &operator=(const CkUrlW &);

    public:
	CkUrlW(void);
	virtual ~CkUrlW(void);

	

	static CkUrlW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// Contains any text following a fragment marker (#) in the URL, excluding the
	// fragment marker. Given the URI http://www.contoso.com/index.htm#main, the
	// fragment is "main".
	void get_Frag(CkString &str);
	// Contains any text following a fragment marker (#) in the URL, excluding the
	// fragment marker. Given the URI http://www.contoso.com/index.htm#main, the
	// fragment is "main".
	const wchar_t *frag(void);

	// The DNS host name or IP address part of the URL. For example, if the URL is
	// "http://www.contoso.com:8080/", the Host is "www.contoso.com". If the URL is
	// "https://192.168.1.124/test.html", the Host is "192.168.1.124".
	void get_Host(CkString &str);
	// The DNS host name or IP address part of the URL. For example, if the URL is
	// "http://www.contoso.com:8080/", the Host is "www.contoso.com". If the URL is
	// "https://192.168.1.124/test.html", the Host is "192.168.1.124".
	const wchar_t *host(void);

	// The type of the host name specified in the URL. Possible values are:
	//     "dns": The host name is a domain name system (DNS) style host name.
	//     "ipv4": The host name is an Internet Protocol (IP) version 4 host address.
	//     "ipv6": The host name is an Internet Protocol (IP) version 6 host address.
	void get_HostType(CkString &str);
	// The type of the host name specified in the URL. Possible values are:
	//     "dns": The host name is a domain name system (DNS) style host name.
	//     "ipv4": The host name is an Internet Protocol (IP) version 4 host address.
	//     "ipv6": The host name is an Internet Protocol (IP) version 6 host address.
	const wchar_t *hostType(void);

	// If the URL contains a login and password, this is the login part. For example,
	// if the URL is "http://user:password@www.contoso.com/index.htm ", then the login
	// is "user".
	void get_Login(CkString &str);
	// If the URL contains a login and password, this is the login part. For example,
	// if the URL is "http://user:password@www.contoso.com/index.htm ", then the login
	// is "user".
	const wchar_t *login(void);

	// If the URL contains a login and password, this is the password part. For
	// example, if the URL is "http://user:password@www.contoso.com/index.htm ", then
	// the password is "password".
	void get_Password(CkString &str);
	// If the URL contains a login and password, this is the password part. For
	// example, if the URL is "http://user:password@www.contoso.com/index.htm ", then
	// the password is "password".
	const wchar_t *password(void);

	// The path (and params) part of the URL, excluding the query and fragment. If the
	// URL is:
	// "http://www.amazon.com/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3?pf_rd_m=AT
	// VPDKIKX0DER&pf_rd_s=desktop-1", then the path is
	// "/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3".
	void get_Path(CkString &str);
	// The path (and params) part of the URL, excluding the query and fragment. If the
	// URL is:
	// "http://www.amazon.com/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3?pf_rd_m=AT
	// VPDKIKX0DER&pf_rd_s=desktop-1", then the path is
	// "/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3".
	const wchar_t *path(void);

	// The path (and params) part of the URL, including the query params, but excluding
	// the fragment. If the URL is:
	// "http://www.amazon.com/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3?pf_rd_m=AT
	// VPDKIKX0DER&pf_rd_s=desktop-1", then then this property returns
	// "/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3?pf_rd_m=ATVPDKIKX0DER&pf_rd_s=d
	// esktop-1".
	void get_PathWithQueryParams(CkString &str);
	// The path (and params) part of the URL, including the query params, but excluding
	// the fragment. If the URL is:
	// "http://www.amazon.com/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3?pf_rd_m=AT
	// VPDKIKX0DER&pf_rd_s=desktop-1", then then this property returns
	// "/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3?pf_rd_m=ATVPDKIKX0DER&pf_rd_s=d
	// esktop-1".
	const wchar_t *pathWithQueryParams(void);

	// The port number of the URL.
	// 
	// For example, if the URL is "http://www.contoso.com:8080/", the port number is
	// 8080.
	// If the URL is "https://192.168.1.124/test.html", the port number is the default
	// 80.
	// If the URL is "https://www.amazon.com/", the port number is the default SSL/TLS
	// port 443.
	// 
	int get_Port(void);

	// The query part of the URL, excluding the fragment. If the URL is:
	// "http://www.amazon.com/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3?pf_rd_m=AT
	// VPDKIKX0DER&pf_rd_s=desktop-1#frag", then the query is
	// "pf_rd_m=ATVPDKIKX0DER&pf_rd_s=desktop-1".
	void get_Query(CkString &str);
	// The query part of the URL, excluding the fragment. If the URL is:
	// "http://www.amazon.com/gp/product/1476752842/ref=s9_psimh_gw_p14_d0_i3?pf_rd_m=AT
	// VPDKIKX0DER&pf_rd_s=desktop-1#frag", then the query is
	// "pf_rd_m=ATVPDKIKX0DER&pf_rd_s=desktop-1".
	const wchar_t *query(void);

	// true if the URL indicates SSL/TLS, otherwise false. A URL beginning with
	// "https://" indicates SSL/TLS.
	bool get_Ssl(void);



	// ----------------------
	// Methods
	// ----------------------
	// Parses a full URL. After parsing, the various parts of the URL are available iin
	// the properties.
	bool ParseUrl(const wchar_t *url);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
