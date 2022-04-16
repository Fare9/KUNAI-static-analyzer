// CkUpload.h: interface for the CkUpload class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkUpload_H
#define _CkUpload_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkByteData;
class CkTask;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkUpload
class CK_VISIBLE_PUBLIC CkUpload  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkUpload(const CkUpload &);
	CkUpload &operator=(const CkUpload &);

    public:
	CkUpload(void);
	virtual ~CkUpload(void);

	static CkUpload *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	CkBaseProgress *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgress *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	bool get_AbortCurrent(void);
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no length file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	void put_AbortCurrent(bool newVal);

	// If non-zero, limits (throttles) the upload bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	int get_BandwidthThrottleUp(void);
	// If non-zero, limits (throttles) the upload bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	void put_BandwidthThrottleUp(int newVal);

	// The chunk size (in bytes) used by the underlying TCP/IP sockets for uploading
	// files. The default value is 65535.
	int get_ChunkSize(void);
	// The chunk size (in bytes) used by the underlying TCP/IP sockets for uploading
	// files. The default value is 65535.
	void put_ChunkSize(int newVal);

	// The IP address to use for computers with multiple network interfaces or IP
	// addresses. For computers with a single network interface (i.e. most computers),
	// this property should not be set. For multihoming computers, the default IP
	// address is automatically used if this property is not set.
	// 
	// The IP address is a string such as in dotted notation using numbers, not domain
	// names, such as "165.164.55.124".
	// 
	void get_ClientIpAddress(CkString &str);
	// The IP address to use for computers with multiple network interfaces or IP
	// addresses. For computers with a single network interface (i.e. most computers),
	// this property should not be set. For multihoming computers, the default IP
	// address is automatically used if this property is not set.
	// 
	// The IP address is a string such as in dotted notation using numbers, not domain
	// names, such as "165.164.55.124".
	// 
	const char *clientIpAddress(void);
	// The IP address to use for computers with multiple network interfaces or IP
	// addresses. For computers with a single network interface (i.e. most computers),
	// this property should not be set. For multihoming computers, the default IP
	// address is automatically used if this property is not set.
	// 
	// The IP address is a string such as in dotted notation using numbers, not domain
	// names, such as "165.164.55.124".
	// 
	void put_ClientIpAddress(const char *newVal);

	// When true, the request header will included an "Expect: 100-continue" header
	// field. This indicates that the server should respond with an intermediate
	// response of "100 Continue" or "417 Expectation Failed" response based on the
	// information available in the request header. This helps avoid situations such as
	// limits on upload sizes. It allows the server to reject the upload, and then the
	// client can abort prior to uploading the data.
	// 
	// The default value of this property is true.
	// 
	bool get_Expect100Continue(void);
	// When true, the request header will included an "Expect: 100-continue" header
	// field. This indicates that the server should respond with an intermediate
	// response of "100 Continue" or "417 Expectation Failed" response based on the
	// information available in the request header. This helps avoid situations such as
	// limits on upload sizes. It allows the server to reject the upload, and then the
	// client can abort prior to uploading the data.
	// 
	// The default value of this property is true.
	// 
	void put_Expect100Continue(bool newVal);

	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Specifies the time interval in milliseconds between AbortCheck events. An Upload
	// operation can be aborted via the AbortCheck event.
	// 
	int get_HeartbeatMs(void);
	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Specifies the time interval in milliseconds between AbortCheck events. An Upload
	// operation can be aborted via the AbortCheck event.
	// 
	void put_HeartbeatMs(int newVal);

	// The hostname of the HTTP server that is the target of the upload. Do not include
	// "http://" in the hostname. It can be a hostname, such as "www.chilkatsoft.com",
	// or an IP address, such as "168.144.70.227".
	void get_Hostname(CkString &str);
	// The hostname of the HTTP server that is the target of the upload. Do not include
	// "http://" in the hostname. It can be a hostname, such as "www.chilkatsoft.com",
	// or an IP address, such as "168.144.70.227".
	const char *hostname(void);
	// The hostname of the HTTP server that is the target of the upload. Do not include
	// "http://" in the hostname. It can be a hostname, such as "www.chilkatsoft.com",
	// or an IP address, such as "168.144.70.227".
	void put_Hostname(const char *newVal);

	// A timeout in milliseconds. The default value is 30000. If the upload hangs (i.e.
	// progress halts) for more than this time, the component will abort the upload.
	// (It will timeout.)
	int get_IdleTimeoutMs(void);
	// A timeout in milliseconds. The default value is 30000. If the upload hangs (i.e.
	// progress halts) for more than this time, the component will abort the upload.
	// (It will timeout.)
	void put_IdleTimeoutMs(int newVal);

	// The HTTP login for sites requiring authentication. Chilkat Upload supports Basic
	// HTTP authentication.
	void get_Login(CkString &str);
	// The HTTP login for sites requiring authentication. Chilkat Upload supports Basic
	// HTTP authentication.
	const char *login(void);
	// The HTTP login for sites requiring authentication. Chilkat Upload supports Basic
	// HTTP authentication.
	void put_Login(const char *newVal);

	// After an upload has completed, this property contains the number of bytes sent.
	// During asynchronous uploads, this property contains the current number of bytes
	// sent while the upload is in progress.
	unsigned long get_NumBytesSent(void);

	// The HTTP password for sites requiring authentication. Chilkat Upload supports
	// Basic HTTP authentication.
	void get_Password(CkString &str);
	// The HTTP password for sites requiring authentication. Chilkat Upload supports
	// Basic HTTP authentication.
	const char *password(void);
	// The HTTP password for sites requiring authentication. Chilkat Upload supports
	// Basic HTTP authentication.
	void put_Password(const char *newVal);

	// The path part of the upload URL. Some examples:
	// 
	// If the upload target (i.e. consumer) URL is:
	// http://www.freeaspupload.net/freeaspupload/testUpload.asp, then
	// 
	//     Hostname = "www.freeaspupload.net" Path = "/freeaspupload/testUpload.asp"
	// 
	// If the upload target URL is
	// http://www.chilkatsoft.com/cgi-bin/ConsumeUpload.exe, then
	// 
	//     Hostname = "www.chilkatsoft.com" Path = "/cgi-bin/ConsumeUpload.exe"
	// 
	void get_Path(CkString &str);
	// The path part of the upload URL. Some examples:
	// 
	// If the upload target (i.e. consumer) URL is:
	// http://www.freeaspupload.net/freeaspupload/testUpload.asp, then
	// 
	//     Hostname = "www.freeaspupload.net" Path = "/freeaspupload/testUpload.asp"
	// 
	// If the upload target URL is
	// http://www.chilkatsoft.com/cgi-bin/ConsumeUpload.exe, then
	// 
	//     Hostname = "www.chilkatsoft.com" Path = "/cgi-bin/ConsumeUpload.exe"
	// 
	const char *path(void);
	// The path part of the upload URL. Some examples:
	// 
	// If the upload target (i.e. consumer) URL is:
	// http://www.freeaspupload.net/freeaspupload/testUpload.asp, then
	// 
	//     Hostname = "www.freeaspupload.net" Path = "/freeaspupload/testUpload.asp"
	// 
	// If the upload target URL is
	// http://www.chilkatsoft.com/cgi-bin/ConsumeUpload.exe, then
	// 
	//     Hostname = "www.chilkatsoft.com" Path = "/cgi-bin/ConsumeUpload.exe"
	// 
	void put_Path(const char *newVal);

	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Sets the value to be defined as 100% complete for the purpose of PercentDone
	// event callbacks. The defaut value of 100 means that at most 100 event
	// PercentDone callbacks will occur in a method that (1) is event enabled and (2)
	// is such that it is possible to measure progress as a percentage completed. This
	// property may be set to larger numbers to get more fine-grained PercentDone
	// callbacks. For example, setting this property equal to 1000 will provide
	// callbacks with .1 percent granularity. For example, a value of 453 would
	// indicate 45.3% competed. This property is clamped to a minimum value of 10, and
	// a maximum value of 100000.
	// 
	int get_PercentDoneScale(void);
	// This property is only valid in programming environment and languages that allow
	// for event callbacks.
	// 
	// Sets the value to be defined as 100% complete for the purpose of PercentDone
	// event callbacks. The defaut value of 100 means that at most 100 event
	// PercentDone callbacks will occur in a method that (1) is event enabled and (2)
	// is such that it is possible to measure progress as a percentage completed. This
	// property may be set to larger numbers to get more fine-grained PercentDone
	// callbacks. For example, setting this property equal to 1000 will provide
	// callbacks with .1 percent granularity. For example, a value of 453 would
	// indicate 45.3% competed. This property is clamped to a minimum value of 10, and
	// a maximum value of 100000.
	// 
	void put_PercentDoneScale(int newVal);

	// Contains the current percentage completion (0 to 100) while an asynchronous
	// upload is in progress.
	unsigned long get_PercentUploaded(void);

	// The port number of the upload target (i.e. consumer) URL. The default value is
	// 80. If SSL is used, this should be set to 443 (typically).
	int get_Port(void);
	// The port number of the upload target (i.e. consumer) URL. The default value is
	// 80. If SSL is used, this should be set to 443 (typically).
	void put_Port(int newVal);

	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	bool get_PreferIpv6(void);
	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	void put_PreferIpv6(bool newVal);

	// The domain name of a proxy host if an HTTP proxy is used. Do not include the
	// "http://". The domain name may be a hostname, such as "www.chilkatsoft.com", or
	// an IP address, such as "168.144.70.227".
	void get_ProxyDomain(CkString &str);
	// The domain name of a proxy host if an HTTP proxy is used. Do not include the
	// "http://". The domain name may be a hostname, such as "www.chilkatsoft.com", or
	// an IP address, such as "168.144.70.227".
	const char *proxyDomain(void);
	// The domain name of a proxy host if an HTTP proxy is used. Do not include the
	// "http://". The domain name may be a hostname, such as "www.chilkatsoft.com", or
	// an IP address, such as "168.144.70.227".
	void put_ProxyDomain(const char *newVal);

	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy login.
	void get_ProxyLogin(CkString &str);
	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy login.
	const char *proxyLogin(void);
	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy login.
	void put_ProxyLogin(const char *newVal);

	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy password.
	void get_ProxyPassword(CkString &str);
	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy password.
	const char *proxyPassword(void);
	// If an HTTP proxy is used and it requires authentication, this property specifies
	// the HTTP proxy password.
	void put_ProxyPassword(const char *newVal);

	// The port number of a proxy server if an HTTP proxy is used.
	int get_ProxyPort(void);
	// The port number of a proxy server if an HTTP proxy is used.
	void put_ProxyPort(int newVal);

	// An HTTP upload is nothing more than an HTTP POST that contains the content of
	// the files being uploaded. Just as with any HTTP POST or GET, the server should
	// send an HTTP response that consists of header and body.
	// 
	// This property contains the body part of the HTTP response.
	// 
	void get_ResponseBody(CkByteData &outBytes);

	// Returns the response body as a string.
	void get_ResponseBodyStr(CkString &str);
	// Returns the response body as a string.
	const char *responseBodyStr(void);

	// An HTTP upload is nothing more than an HTTP POST that contains the content of
	// the files being uploaded. Just as with any HTTP POST or GET, the server should
	// send an HTTP response that consists of header and body.
	// 
	// This property contains the header part of the HTTP response.
	// 
	void get_ResponseHeader(CkString &str);
	// An HTTP upload is nothing more than an HTTP POST that contains the content of
	// the files being uploaded. Just as with any HTTP POST or GET, the server should
	// send an HTTP response that consists of header and body.
	// 
	// This property contains the header part of the HTTP response.
	// 
	const char *responseHeader(void);

	// The HTTP response status code of the HTTP response. A list of HTTP status codes
	// can be found here:HTTP Response Status Codes
	// <http://en.wikipedia.org/wiki/List_of_HTTP_status_codes>.
	int get_ResponseStatus(void);

	// Set this to true if the upload is to HTTPS. For example, if the target of the
	// upload is:
	// 
	//     https://www.myuploadtarget.com/consumeUpload.asp
	// 
	// then set:
	// 
	//     Ssl = true
	//     Hostname = "www.myuploadtarget.com"
	//     Path = "/consumeupload.asp"
	//     Port = 443
	//     
	// 
	bool get_Ssl(void);
	// Set this to true if the upload is to HTTPS. For example, if the target of the
	// upload is:
	// 
	//     https://www.myuploadtarget.com/consumeUpload.asp
	// 
	// then set:
	// 
	//     Ssl = true
	//     Hostname = "www.myuploadtarget.com"
	//     Path = "/consumeupload.asp"
	//     Port = 443
	//     
	// 
	void put_Ssl(bool newVal);

	// Provides a means for setting a list of ciphers that are allowed for SSL/TLS
	// connections. The default (empty string) indicates that all implemented ciphers
	// are possible. The TLS ciphers supported in Chilkat v9.5.0.55 and later are:
	// TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_RSA_WITH_AES_256_CBC_SHA256
	// TLS_RSA_WITH_AES_256_GCM_SHA384
	// TLS_RSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_RSA_WITH_AES_128_CBC_SHA256
	// TLS_RSA_WITH_AES_128_GCM_SHA256
	// TLS_RSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_ECDHE_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_MD5
	// TLS_DHE_RSA_WITH_DES_CBC_SHA
	// TLS_RSA_WITH_DES_CBC_SHA
	// To restrict SSL/TLS connections to one or more specific ciphers, set this
	// property to a comma-separated list of ciphers such as
	// "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384".
	// The order should be in terms of preference, with the preferred algorithms listed
	// first. (Note that the client cannot specifically choose the algorithm is picked
	// because it is the server that chooses. The client simply provides the server
	// with a list from which to choose.)
	// 
	// The property can also disallow connections with servers having certificates with
	// RSA keys less than a certain size. By default, server certificates having RSA
	// keys of 512 bits or greater are allowed. Add the keyword "rsa1024" to disallow
	// connections with servers having keys smaller than 1024 bits. Add the keyword
	// "rsa2048" to disallow connections with servers having keys smaller than 2048
	// bits.
	// 
	// Note: Prior to Chilkat v9.5.0.55, it was not possible to explicitly list allowed
	// cipher suites. The deprecated means for indicating allowed ciphers was both
	// incomplete and unprecise. For example, the following keywords could be listed to
	// allow matching ciphers: "aes256-cbc", "aes128-cbc", "3des-cbc", and "rc4". These
	// keywords will still be recognized, but programs should be updated to explicitly
	// list the allowed ciphers.
	// 
	// secure-renegotiation: Starting in Chilkat v9.5.0.55, the keyword
	// "secure-renegotiation" may be added to require that all renegotions be done
	// securely (as per RFC 5746).
	// 
	// best-practices: Starting in Chilkat v9.5.0.55, this property may be set to the
	// single keyword "best-practices". This will allow ciphers based on the current
	// best practices. As new versions of Chilkat are released, the best practices may
	// change. Changes will be noted here. The current best practices are:
	// 
	//     If the server uses an RSA key, it must be 1024 bits or greater.
	//     All renegotations must be secure renegotiations.
	//     All ciphers using RC4, DES, or 3DES are disallowed.
	// 
	// Example: The following string would restrict to 2 specific cipher suites,
	// require RSA keys to be 1024 bits or greater, and require secure renegotiations:
	// "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256, TLS_RSA_WITH_AES_256_CBC_SHA, rsa1024,
	// secure-renegotiation"
	// 
	void get_SslAllowedCiphers(CkString &str);
	// Provides a means for setting a list of ciphers that are allowed for SSL/TLS
	// connections. The default (empty string) indicates that all implemented ciphers
	// are possible. The TLS ciphers supported in Chilkat v9.5.0.55 and later are:
	// TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_RSA_WITH_AES_256_CBC_SHA256
	// TLS_RSA_WITH_AES_256_GCM_SHA384
	// TLS_RSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_RSA_WITH_AES_128_CBC_SHA256
	// TLS_RSA_WITH_AES_128_GCM_SHA256
	// TLS_RSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_ECDHE_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_MD5
	// TLS_DHE_RSA_WITH_DES_CBC_SHA
	// TLS_RSA_WITH_DES_CBC_SHA
	// To restrict SSL/TLS connections to one or more specific ciphers, set this
	// property to a comma-separated list of ciphers such as
	// "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384".
	// The order should be in terms of preference, with the preferred algorithms listed
	// first. (Note that the client cannot specifically choose the algorithm is picked
	// because it is the server that chooses. The client simply provides the server
	// with a list from which to choose.)
	// 
	// The property can also disallow connections with servers having certificates with
	// RSA keys less than a certain size. By default, server certificates having RSA
	// keys of 512 bits or greater are allowed. Add the keyword "rsa1024" to disallow
	// connections with servers having keys smaller than 1024 bits. Add the keyword
	// "rsa2048" to disallow connections with servers having keys smaller than 2048
	// bits.
	// 
	// Note: Prior to Chilkat v9.5.0.55, it was not possible to explicitly list allowed
	// cipher suites. The deprecated means for indicating allowed ciphers was both
	// incomplete and unprecise. For example, the following keywords could be listed to
	// allow matching ciphers: "aes256-cbc", "aes128-cbc", "3des-cbc", and "rc4". These
	// keywords will still be recognized, but programs should be updated to explicitly
	// list the allowed ciphers.
	// 
	// secure-renegotiation: Starting in Chilkat v9.5.0.55, the keyword
	// "secure-renegotiation" may be added to require that all renegotions be done
	// securely (as per RFC 5746).
	// 
	// best-practices: Starting in Chilkat v9.5.0.55, this property may be set to the
	// single keyword "best-practices". This will allow ciphers based on the current
	// best practices. As new versions of Chilkat are released, the best practices may
	// change. Changes will be noted here. The current best practices are:
	// 
	//     If the server uses an RSA key, it must be 1024 bits or greater.
	//     All renegotations must be secure renegotiations.
	//     All ciphers using RC4, DES, or 3DES are disallowed.
	// 
	// Example: The following string would restrict to 2 specific cipher suites,
	// require RSA keys to be 1024 bits or greater, and require secure renegotiations:
	// "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256, TLS_RSA_WITH_AES_256_CBC_SHA, rsa1024,
	// secure-renegotiation"
	// 
	const char *sslAllowedCiphers(void);
	// Provides a means for setting a list of ciphers that are allowed for SSL/TLS
	// connections. The default (empty string) indicates that all implemented ciphers
	// are possible. The TLS ciphers supported in Chilkat v9.5.0.55 and later are:
	// TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_256_GCM_SHA384
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA
	// TLS_RSA_WITH_AES_256_CBC_SHA256
	// TLS_RSA_WITH_AES_256_GCM_SHA384
	// TLS_RSA_WITH_AES_256_CBC_SHA
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA256
	// TLS_DHE_RSA_WITH_AES_128_GCM_SHA256
	// TLS_DHE_RSA_WITH_AES_128_CBC_SHA
	// TLS_RSA_WITH_AES_128_CBC_SHA256
	// TLS_RSA_WITH_AES_128_GCM_SHA256
	// TLS_RSA_WITH_AES_128_CBC_SHA
	// TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_RSA_WITH_3DES_EDE_CBC_SHA
	// TLS_ECDHE_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_SHA
	// TLS_RSA_WITH_RC4_128_MD5
	// TLS_DHE_RSA_WITH_DES_CBC_SHA
	// TLS_RSA_WITH_DES_CBC_SHA
	// To restrict SSL/TLS connections to one or more specific ciphers, set this
	// property to a comma-separated list of ciphers such as
	// "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384, TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384".
	// The order should be in terms of preference, with the preferred algorithms listed
	// first. (Note that the client cannot specifically choose the algorithm is picked
	// because it is the server that chooses. The client simply provides the server
	// with a list from which to choose.)
	// 
	// The property can also disallow connections with servers having certificates with
	// RSA keys less than a certain size. By default, server certificates having RSA
	// keys of 512 bits or greater are allowed. Add the keyword "rsa1024" to disallow
	// connections with servers having keys smaller than 1024 bits. Add the keyword
	// "rsa2048" to disallow connections with servers having keys smaller than 2048
	// bits.
	// 
	// Note: Prior to Chilkat v9.5.0.55, it was not possible to explicitly list allowed
	// cipher suites. The deprecated means for indicating allowed ciphers was both
	// incomplete and unprecise. For example, the following keywords could be listed to
	// allow matching ciphers: "aes256-cbc", "aes128-cbc", "3des-cbc", and "rc4". These
	// keywords will still be recognized, but programs should be updated to explicitly
	// list the allowed ciphers.
	// 
	// secure-renegotiation: Starting in Chilkat v9.5.0.55, the keyword
	// "secure-renegotiation" may be added to require that all renegotions be done
	// securely (as per RFC 5746).
	// 
	// best-practices: Starting in Chilkat v9.5.0.55, this property may be set to the
	// single keyword "best-practices". This will allow ciphers based on the current
	// best practices. As new versions of Chilkat are released, the best practices may
	// change. Changes will be noted here. The current best practices are:
	// 
	//     If the server uses an RSA key, it must be 1024 bits or greater.
	//     All renegotations must be secure renegotiations.
	//     All ciphers using RC4, DES, or 3DES are disallowed.
	// 
	// Example: The following string would restrict to 2 specific cipher suites,
	// require RSA keys to be 1024 bits or greater, and require secure renegotiations:
	// "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256, TLS_RSA_WITH_AES_256_CBC_SHA, rsa1024,
	// secure-renegotiation"
	// 
	void put_SslAllowedCiphers(const char *newVal);

	// Selects the secure protocol to be used for secure (SSL/TLS) connections.
	// Possible values are:
	// 
	//     default
	//     TLS 1.3
	//     TLS 1.2
	//     TLS 1.1
	//     TLS 1.0
	//     SSL 3.0
	//     TLS 1.3 or higher
	//     TLS 1.2 or higher
	//     TLS 1.1 or higher
	//     TLS 1.0 or higher
	//     
	// 
	// The default value is "default" which will choose the, which allows for the
	// protocol to be selected dynamically at runtime based on the requirements of the
	// server. Choosing an exact protocol will cause the connection to fail unless that
	// exact protocol is negotiated. It is better to choose "X or higher" than an exact
	// protocol. The "default" is effectively "SSL 3.0 or higher".
	void get_SslProtocol(CkString &str);
	// Selects the secure protocol to be used for secure (SSL/TLS) connections.
	// Possible values are:
	// 
	//     default
	//     TLS 1.3
	//     TLS 1.2
	//     TLS 1.1
	//     TLS 1.0
	//     SSL 3.0
	//     TLS 1.3 or higher
	//     TLS 1.2 or higher
	//     TLS 1.1 or higher
	//     TLS 1.0 or higher
	//     
	// 
	// The default value is "default" which will choose the, which allows for the
	// protocol to be selected dynamically at runtime based on the requirements of the
	// server. Choosing an exact protocol will cause the connection to fail unless that
	// exact protocol is negotiated. It is better to choose "X or higher" than an exact
	// protocol. The "default" is effectively "SSL 3.0 or higher".
	const char *sslProtocol(void);
	// Selects the secure protocol to be used for secure (SSL/TLS) connections.
	// Possible values are:
	// 
	//     default
	//     TLS 1.3
	//     TLS 1.2
	//     TLS 1.1
	//     TLS 1.0
	//     SSL 3.0
	//     TLS 1.3 or higher
	//     TLS 1.2 or higher
	//     TLS 1.1 or higher
	//     TLS 1.0 or higher
	//     
	// 
	// The default value is "default" which will choose the, which allows for the
	// protocol to be selected dynamically at runtime based on the requirements of the
	// server. Choosing an exact protocol will cause the connection to fail unless that
	// exact protocol is negotiated. It is better to choose "X or higher" than an exact
	// protocol. The "default" is effectively "SSL 3.0 or higher".
	void put_SslProtocol(const char *newVal);

	// Specifies a set of pins for Public Key Pinning for TLS connections. This
	// property lists the expected SPKI fingerprints for the server certificates. If
	// the server's certificate (sent during the TLS handshake) does not match any of
	// the SPKI fingerprints, then the TLS handshake is aborted and the connection
	// fails. The format of this string property is as follows:
	// hash_algorithm, encoding, SPKI_fingerprint_1, SPKI_fingerprint_2, ...
	// For example, the following string specifies a single sha256 base64-encoded SPKI
	// fingerprint:
	// "sha256, base64, lKg1SIqyhPSK19tlPbjl8s02yChsVTDklQpkMCHvsTE="
	// This example specifies two SPKI fingerprints:
	// "sha256, base64, 4t37LpnGmrMEAG8HEz9yIrnvJV2euVRwCLb9EH5WZyI=, 68b0G5iqMvWVWvUCjMuhLEyekM5729PadtnU5tdXZKs="
	// Any of the following hash algorithms are allowed:.sha1, sha256, sha384, sha512,
	// md2, md5, haval, ripemd128, ripemd160,ripemd256, or ripemd320.
	// 
	// The following encodings are allowed: base64, hex, and any of the encodings
	// indicated in the link below.
	// 
	void get_TlsPinSet(CkString &str);
	// Specifies a set of pins for Public Key Pinning for TLS connections. This
	// property lists the expected SPKI fingerprints for the server certificates. If
	// the server's certificate (sent during the TLS handshake) does not match any of
	// the SPKI fingerprints, then the TLS handshake is aborted and the connection
	// fails. The format of this string property is as follows:
	// hash_algorithm, encoding, SPKI_fingerprint_1, SPKI_fingerprint_2, ...
	// For example, the following string specifies a single sha256 base64-encoded SPKI
	// fingerprint:
	// "sha256, base64, lKg1SIqyhPSK19tlPbjl8s02yChsVTDklQpkMCHvsTE="
	// This example specifies two SPKI fingerprints:
	// "sha256, base64, 4t37LpnGmrMEAG8HEz9yIrnvJV2euVRwCLb9EH5WZyI=, 68b0G5iqMvWVWvUCjMuhLEyekM5729PadtnU5tdXZKs="
	// Any of the following hash algorithms are allowed:.sha1, sha256, sha384, sha512,
	// md2, md5, haval, ripemd128, ripemd160,ripemd256, or ripemd320.
	// 
	// The following encodings are allowed: base64, hex, and any of the encodings
	// indicated in the link below.
	// 
	const char *tlsPinSet(void);
	// Specifies a set of pins for Public Key Pinning for TLS connections. This
	// property lists the expected SPKI fingerprints for the server certificates. If
	// the server's certificate (sent during the TLS handshake) does not match any of
	// the SPKI fingerprints, then the TLS handshake is aborted and the connection
	// fails. The format of this string property is as follows:
	// hash_algorithm, encoding, SPKI_fingerprint_1, SPKI_fingerprint_2, ...
	// For example, the following string specifies a single sha256 base64-encoded SPKI
	// fingerprint:
	// "sha256, base64, lKg1SIqyhPSK19tlPbjl8s02yChsVTDklQpkMCHvsTE="
	// This example specifies two SPKI fingerprints:
	// "sha256, base64, 4t37LpnGmrMEAG8HEz9yIrnvJV2euVRwCLb9EH5WZyI=, 68b0G5iqMvWVWvUCjMuhLEyekM5729PadtnU5tdXZKs="
	// Any of the following hash algorithms are allowed:.sha1, sha256, sha384, sha512,
	// md2, md5, haval, ripemd128, ripemd160,ripemd256, or ripemd320.
	// 
	// The following encodings are allowed: base64, hex, and any of the encodings
	// indicated in the link below.
	// 
	void put_TlsPinSet(const char *newVal);

	// The total size of the upload (in bytes). This property will become set at the
	// beginning of an asynchronous upload. A program may monitor asynchronous uploads
	// by tracking both NumBytesSent and PercentUploaded.
	// 
	// This property is also set during synchronous uploads.
	// 
	unsigned long get_TotalUploadSize(void);

	// Set to true when an asynchronous upload is started. When the asynchronous
	// upload is complete, this property becomes equal to false. A program will
	// typically begin an asynchronous upload by calling BeginUpload, and then
	// periodically checking the value of this property to determine when the upload is
	// complete.
	bool get_UploadInProgress(void);

	// Set to true (success) or false (failed) after an asynchronous upload
	// completes or aborts due to failure. When a program does an asynchronous upload,
	// it will wait until UploadInProgress becomes false. It will then check the
	// value of this property to determine if the upload was successful or not.
	bool get_UploadSuccess(void);



	// ----------------------
	// Methods
	// ----------------------
	// May be called during an asynchronous upload to abort.
	void AbortUpload(void);


	// Adds a custom HTTP header to the HTTP upload.
	void AddCustomHeader(const char *name, const char *value);


	// Adds a file to the list of files to be uploaded in the next call to
	// BlockingUpload, BeginUpload, or UploadToMemory. To upload more than one file,
	// call this method once for each file to be uploaded.
	void AddFileReference(const char *name, const char *filename);


	// Adds a custom HTTP request parameter to the upload.
	void AddParam(const char *name, const char *value);


	// Starts an asynchronous upload. Only one asynchronous upload may be in progress
	// at a time. To achieve multiple asynchronous uploads, use multiple instances of
	// the Chilkat Upload object. Each object instance is capable of managing a single
	// asynchronous upload.
	// 
	// When this method is called, a background thread is started and the asynchronous
	// upload runs in the background. The upload may be aborted at any time by calling
	// AbortUpload. The upload is completed (or failed) when UploadInProgress becomes
	// false. At that point, the UploadSuccess property may be checked to determine
	// success (true) or failure (false).
	// 
	bool BeginUpload(void);


	// Uploads files to a web server using HTTP. The files to be uploaded are indicated
	// by calling AddFileReference once for each file (prior to calling this method).
	bool BlockingUpload(void);

	// Uploads files to a web server using HTTP. The files to be uploaded are indicated
	// by calling AddFileReference once for each file (prior to calling this method).
	CkTask *BlockingUploadAsync(void);


	// Clears the internal list of files created by calls to AddFileReference.
	void ClearFileReferences(void);


	// Clears the internal list of params created by calls to AddParam.
	void ClearParams(void);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// A convenience method for putting the calling process to sleep for N
	// milliseconds. It is provided here because some programming languages do not
	// provide this capability, and sleeping for short increments of time is helpful
	// when doing asynchronous uploads.
	void SleepMs(int millisec);


	// Writes the complete HTTP POST to memory. The POST contains the HTTP header, any
	// custom params added by calling AddParam, and each file to be uploaded. This is
	// helpful in debugging. It allows you to see the exact HTTP POST sent to the
	// server if BlockingUpload or BeginUpload is called.
	bool UploadToMemory(CkByteData &outData);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
