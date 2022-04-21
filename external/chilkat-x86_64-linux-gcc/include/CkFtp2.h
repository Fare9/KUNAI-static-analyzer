// CkFtp2.h: interface for the CkFtp2 class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkFtp2_H
#define _CkFtp2_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkByteData;
class CkDateTime;
class CkBinData;
class CkStringBuilder;
class CkStream;
class CkCert;
class CkSecureString;
class CkFtp2Progress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkFtp2
class CK_VISIBLE_PUBLIC CkFtp2  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkFtp2(const CkFtp2 &);
	CkFtp2 &operator=(const CkFtp2 &);

    public:
	CkFtp2(void);
	virtual ~CkFtp2(void);

	static CkFtp2 *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	CkFtp2Progress *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkFtp2Progress *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no lengthy file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	bool get_AbortCurrent(void);
	// When set to true, causes the currently running method to abort. Methods that
	// always finish quickly (i.e.have no lengthy file operations or network
	// communications) are not affected. If no method is running, then this property is
	// automatically reset to false when the next method is called. When the abort
	// occurs, this property is reset to false. Both synchronous and asynchronous
	// method calls can be aborted. (A synchronous method call could be aborted by
	// setting this property from a separate thread.)
	void put_AbortCurrent(bool newVal);

	// Some FTP servers require an Account name in addition to login/password. This
	// property can be set for those servers with this requirement.
	void get_Account(CkString &str);
	// Some FTP servers require an Account name in addition to login/password. This
	// property can be set for those servers with this requirement.
	const char *account(void);
	// Some FTP servers require an Account name in addition to login/password. This
	// property can be set for those servers with this requirement.
	void put_Account(const char *newVal);

	// When Active (i.e. PORT) mode is used (opposite of Passive), the client-side is
	// responsible for choosing a random port for each data connection. (Note: In the
	// FTP protocol, each data transfer occurs on a separate TCP/IP connection.
	// Commands are sent over the control channel (port 21 for non-SSL, port 990 for
	// SSL).)
	// 
	// This property, along with ActivePortRangeStart, allows the client to specify a
	// range of ports for data connections.
	// 
	int get_ActivePortRangeEnd(void);
	// When Active (i.e. PORT) mode is used (opposite of Passive), the client-side is
	// responsible for choosing a random port for each data connection. (Note: In the
	// FTP protocol, each data transfer occurs on a separate TCP/IP connection.
	// Commands are sent over the control channel (port 21 for non-SSL, port 990 for
	// SSL).)
	// 
	// This property, along with ActivePortRangeStart, allows the client to specify a
	// range of ports for data connections.
	// 
	void put_ActivePortRangeEnd(int newVal);

	// This property, along with ActivePortRangeEnd, allows the client to specify a
	// range of ports for data connections when in Active mode.
	int get_ActivePortRangeStart(void);
	// This property, along with ActivePortRangeEnd, allows the client to specify a
	// range of ports for data connections when in Active mode.
	void put_ActivePortRangeStart(int newVal);

	// If set to a non-zero value, causes an ALLO command, with this size as the
	// parameter, to be automatically sent when uploading files to an FTP server.
	// 
	// This command could be required by some servers to reserve sufficient storage
	// space to accommodate the new file to be transferred.
	// 
	unsigned long get_AllocateSize(void);
	// If set to a non-zero value, causes an ALLO command, with this size as the
	// parameter, to be automatically sent when uploading files to an FTP server.
	// 
	// This command could be required by some servers to reserve sufficient storage
	// space to accommodate the new file to be transferred.
	// 
	void put_AllocateSize(unsigned long newVal);

	// If true, then uses the MLSD command to fetch directory listings when the FTP
	// server supports MLSD. This property is true by default.
	// 
	// When MLSD is used, the GetPermissions method will return the "perm fact" for a
	// given file or directory. This is a different format than the more commonly
	// recognized UNIX permissions string. Note: MLSD provides more accurate and
	// dependable file listings, especially for last-mod date/time information. If
	// usage of the MLSD command is turned off, it may adversely affect the quality and
	// availability of other information.
	// 
	bool get_AllowMlsd(void);
	// If true, then uses the MLSD command to fetch directory listings when the FTP
	// server supports MLSD. This property is true by default.
	// 
	// When MLSD is used, the GetPermissions method will return the "perm fact" for a
	// given file or directory. This is a different format than the more commonly
	// recognized UNIX permissions string. Note: MLSD provides more accurate and
	// dependable file listings, especially for last-mod date/time information. If
	// usage of the MLSD command is turned off, it may adversely affect the quality and
	// availability of other information.
	// 
	void put_AllowMlsd(bool newVal);

	// The number of bytes received during an asynchronous FTP download. This property
	// is updated in real-time and an application may periodically fetch and display
	// it's value while the download is in progress.
	unsigned long get_AsyncBytesReceived(void);

	// Same as AsyncBytesReceived, but returns the value as a 64-bit integer.
	__int64 get_AsyncBytesReceived64(void);

	// The number of bytes received during an asynchronous FTP download. This property
	// is updated in real-time and an application may periodically fetch and display
	// it's value while the download is in progress.
	void get_AsyncBytesReceivedStr(CkString &str);
	// The number of bytes received during an asynchronous FTP download. This property
	// is updated in real-time and an application may periodically fetch and display
	// it's value while the download is in progress.
	const char *asyncBytesReceivedStr(void);

	// The number of bytes sent during an asynchronous FTP upload. This property is
	// updated in real-time and an application may periodically fetch and display it's
	// value while the upload is in progress.
	unsigned long get_AsyncBytesSent(void);

	// Same as AsyncBytesSent, but returns the value as a 64-bit integer.
	__int64 get_AsyncBytesSent64(void);

	// The number of bytes sent during an asynchronous FTP upload. This string property
	// is updated in real-time and an application may periodically fetch and display
	// it's value while the upload is in progress.
	void get_AsyncBytesSentStr(CkString &str);
	// The number of bytes sent during an asynchronous FTP upload. This string property
	// is updated in real-time and an application may periodically fetch and display
	// it's value while the upload is in progress.
	const char *asyncBytesSentStr(void);

	// Same as AuthTls, except the command sent to the FTP server is "AUTH SSL" instead
	// of "AUTH TLS". Most FTP servers accept either. AuthTls is more commonly used. If
	// a particular server has trouble with AuthTls, try AuthSsl instead.
	bool get_AuthSsl(void);
	// Same as AuthTls, except the command sent to the FTP server is "AUTH SSL" instead
	// of "AUTH TLS". Most FTP servers accept either. AuthTls is more commonly used. If
	// a particular server has trouble with AuthTls, try AuthSsl instead.
	void put_AuthSsl(bool newVal);

	// Set this to true to switch to a TLS encrypted channel. This property should be
	// set prior to connecting. If this property is set, the port typically remains at
	// it's default (21) and the Ssl property should *not* be set. When AuthTls is
	// used, all control and data transmissions are encrypted. If your FTP client is
	// behind a network-address-translating router, you may need to call
	// ClearControlChannel after connecting and authenticating (i.e. after calling the
	// Connect method). This keeps all data transmissions encrypted, but clears the
	// control channel so that commands are sent unencrypted, thus allowing the router
	// to translate network IP numbers in FTP commands.
	bool get_AuthTls(void);
	// Set this to true to switch to a TLS encrypted channel. This property should be
	// set prior to connecting. If this property is set, the port typically remains at
	// it's default (21) and the Ssl property should *not* be set. When AuthTls is
	// used, all control and data transmissions are encrypted. If your FTP client is
	// behind a network-address-translating router, you may need to call
	// ClearControlChannel after connecting and authenticating (i.e. after calling the
	// Connect method). This keeps all data transmissions encrypted, but clears the
	// control channel so that commands are sent unencrypted, thus allowing the router
	// to translate network IP numbers in FTP commands.
	void put_AuthTls(bool newVal);

	// When true (which is the default value), a "FEAT" command is automatically sent
	// to the FTP server immediately after connecting. This allows the Chilkat FTP2
	// component to know more about the server's capabilities and automatically adjust
	// any applicable internal settings based on the response. In rare cases, some FTP
	// servers reject the "FEAT" command and close the connection. Usually, if an FTP
	// server does not implement FEAT, a harmless "command not understood" response is
	// returned.
	// 
	// Set this property to false to prevent the FEAT command from being sent.
	// 
	bool get_AutoFeat(void);
	// When true (which is the default value), a "FEAT" command is automatically sent
	// to the FTP server immediately after connecting. This allows the Chilkat FTP2
	// component to know more about the server's capabilities and automatically adjust
	// any applicable internal settings based on the response. In rare cases, some FTP
	// servers reject the "FEAT" command and close the connection. Usually, if an FTP
	// server does not implement FEAT, a harmless "command not understood" response is
	// returned.
	// 
	// Set this property to false to prevent the FEAT command from being sent.
	// 
	void put_AutoFeat(bool newVal);

	// If true, then the following will occur when a connection is made to an FTP
	// server:
	// 
	// 1) If the Port property = 990, then sets AuthTls = false, AuthSsl = false,
	// and Ssl = true
	// 2) If the Port property = 21, sets Ssl = false
	// 
	// The default value of this property is true.
	// 
	bool get_AutoFix(void);
	// If true, then the following will occur when a connection is made to an FTP
	// server:
	// 
	// 1) If the Port property = 990, then sets AuthTls = false, AuthSsl = false,
	// and Ssl = true
	// 2) If the Port property = 21, sets Ssl = false
	// 
	// The default value of this property is true.
	// 
	void put_AutoFix(bool newVal);

	// Forces the component to retrieve each file's size prior to downloading for the
	// purpose of monitoring percentage completion progress. For many FTP servers, this
	// is not required and therefore for performance reasons this property defaults to
	// false.
	bool get_AutoGetSizeForProgress(void);
	// Forces the component to retrieve each file's size prior to downloading for the
	// purpose of monitoring percentage completion progress. For many FTP servers, this
	// is not required and therefore for performance reasons this property defaults to
	// false.
	void put_AutoGetSizeForProgress(bool newVal);

	// When true (which is the default value), then an "OPTS UTF8 ON" command is
	// automatically sent when connecting/authenticating if it is discovered via the
	// FEAT command that the UTF8 option is supported.
	// 
	// Set this property to false to prevent the "OPTS UTF8 ON" command from being
	// sent.
	// 
	bool get_AutoOptsUtf8(void);
	// When true (which is the default value), then an "OPTS UTF8 ON" command is
	// automatically sent when connecting/authenticating if it is discovered via the
	// FEAT command that the UTF8 option is supported.
	// 
	// Set this property to false to prevent the "OPTS UTF8 ON" command from being
	// sent.
	// 
	void put_AutoOptsUtf8(bool newVal);

	// If true then the UseEpsv property is automatically set upon connecting to the
	// FTP server. The default value of this property is false.
	// 
	// If the AutoFeat property is true, and if the AutoSetUseEpsv property is
	// true, then the FTP server's features are automatically queried when
	// connecting. In this case, the UseEpsv property is automatically set to true if
	// the FTP server supports EPSV.
	// 
	// Important: EPSV can cause problems with some deep-inspection firewalls. If a
	// passive data connection cannot be established, make sure to test with both the
	// AutoSetUseEpsv and UseEpsv properties set equal to false.
	// 
	bool get_AutoSetUseEpsv(void);
	// If true then the UseEpsv property is automatically set upon connecting to the
	// FTP server. The default value of this property is false.
	// 
	// If the AutoFeat property is true, and if the AutoSetUseEpsv property is
	// true, then the FTP server's features are automatically queried when
	// connecting. In this case, the UseEpsv property is automatically set to true if
	// the FTP server supports EPSV.
	// 
	// Important: EPSV can cause problems with some deep-inspection firewalls. If a
	// passive data connection cannot be established, make sure to test with both the
	// AutoSetUseEpsv and UseEpsv properties set equal to false.
	// 
	void put_AutoSetUseEpsv(bool newVal);

	// When true (which is the default value), a "SYST" command is automatically sent
	// to the FTP server immediately after connecting. This allows the Chilkat FTP2
	// component to know more about the server and automatically adjust any applicable
	// internal settings based on the response. If the SYST command causes trouble
	// (which is rare), this behavior can be turned off by setting this property equal
	// to false.
	bool get_AutoSyst(void);
	// When true (which is the default value), a "SYST" command is automatically sent
	// to the FTP server immediately after connecting. This allows the Chilkat FTP2
	// component to know more about the server and automatically adjust any applicable
	// internal settings based on the response. If the SYST command causes trouble
	// (which is rare), this behavior can be turned off by setting this property equal
	// to false.
	void put_AutoSyst(bool newVal);

	// Many FTP servers support the XCRC command. The Chilkat FTP component will
	// automatically know if XCRC is supported because it automatically sends a FEAT
	// command to the server immediately after connecting.
	// 
	// If this property is set to true, then all uploads will be automatically
	// verified by sending an XCRC command immediately after the transfer completes. If
	// the CRC is not verified, the upload method (such as PutFile) will return a
	// failed status.
	// 
	// To prevent XCRC checking, set this property to false.
	// 
	bool get_AutoXcrc(void);
	// Many FTP servers support the XCRC command. The Chilkat FTP component will
	// automatically know if XCRC is supported because it automatically sends a FEAT
	// command to the server immediately after connecting.
	// 
	// If this property is set to true, then all uploads will be automatically
	// verified by sending an XCRC command immediately after the transfer completes. If
	// the CRC is not verified, the upload method (such as PutFile) will return a
	// failed status.
	// 
	// To prevent XCRC checking, set this property to false.
	// 
	void put_AutoXcrc(bool newVal);

	// If set to a non-zero value, the FTP2 component will bandwidth throttle all
	// downloads to this value.
	// 
	// The default value of this property is 0. The value should be specified in
	// bytes/second.
	// 
	// Note: It is difficult to throttle very small downloads. (For example, how do you
	// bandwidth throttle a 1-byte download???) As the downloaded file size gets
	// larger, the transfer rate will better approximate this property's setting.
	// 
	// Also note: When downloading, the FTP server has no knowledge of the client's
	// desire for throttling, and is always sending data as fast as possible. (There's
	// nothing in the FTP protocol to request throttling.) Therefore, any throttling
	// for a download on the client side is simply to allow system socket buffers
	// (outgoing buffers on the sender, and incoming buffers on the client) to fill to
	// 100% capacity, and this also poses the threat of causing a data connection to
	// become broken. It's probably not worthwhile to attempt to throttle downloads. It
	// may have been better that this property never existed.
	// 
	int get_BandwidthThrottleDown(void);
	// If set to a non-zero value, the FTP2 component will bandwidth throttle all
	// downloads to this value.
	// 
	// The default value of this property is 0. The value should be specified in
	// bytes/second.
	// 
	// Note: It is difficult to throttle very small downloads. (For example, how do you
	// bandwidth throttle a 1-byte download???) As the downloaded file size gets
	// larger, the transfer rate will better approximate this property's setting.
	// 
	// Also note: When downloading, the FTP server has no knowledge of the client's
	// desire for throttling, and is always sending data as fast as possible. (There's
	// nothing in the FTP protocol to request throttling.) Therefore, any throttling
	// for a download on the client side is simply to allow system socket buffers
	// (outgoing buffers on the sender, and incoming buffers on the client) to fill to
	// 100% capacity, and this also poses the threat of causing a data connection to
	// become broken. It's probably not worthwhile to attempt to throttle downloads. It
	// may have been better that this property never existed.
	// 
	void put_BandwidthThrottleDown(int newVal);

	// If set to a non-zero value, the FTP2 component will bandwidth throttle all
	// uploads to this value.
	// 
	// The default value of this property is 0. The value should be specified in
	// bytes/second.
	// 
	// Note: It is difficult to throttle very small uploads. (For example, how do you
	// bandwidth throttle a 1-byte upload???) As the uploaded file size gets larger,
	// the transfer rate will better approximate this property's setting.
	// 
	int get_BandwidthThrottleUp(void);
	// If set to a non-zero value, the FTP2 component will bandwidth throttle all
	// uploads to this value.
	// 
	// The default value of this property is 0. The value should be specified in
	// bytes/second.
	// 
	// Note: It is difficult to throttle very small uploads. (For example, how do you
	// bandwidth throttle a 1-byte upload???) As the uploaded file size gets larger,
	// the transfer rate will better approximate this property's setting.
	// 
	void put_BandwidthThrottleUp(int newVal);

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

	// Indicates the charset to be used for commands sent to the FTP server. The
	// command charset must match what the FTP server is expecting in order to
	// communicate non-English characters correctly. The default value of this property
	// is "ansi".
	// 
	// This property may be updated to "utf-8" after connecting because a "FEAT"
	// command is automatically sent to get the features of the FTP server. If UTF8 is
	// indicated as a feature, then this property is automatically changed to "utf-8".
	// 
	void get_CommandCharset(CkString &str);
	// Indicates the charset to be used for commands sent to the FTP server. The
	// command charset must match what the FTP server is expecting in order to
	// communicate non-English characters correctly. The default value of this property
	// is "ansi".
	// 
	// This property may be updated to "utf-8" after connecting because a "FEAT"
	// command is automatically sent to get the features of the FTP server. If UTF8 is
	// indicated as a feature, then this property is automatically changed to "utf-8".
	// 
	const char *commandCharset(void);
	// Indicates the charset to be used for commands sent to the FTP server. The
	// command charset must match what the FTP server is expecting in order to
	// communicate non-English characters correctly. The default value of this property
	// is "ansi".
	// 
	// This property may be updated to "utf-8" after connecting because a "FEAT"
	// command is automatically sent to get the features of the FTP server. If UTF8 is
	// indicated as a feature, then this property is automatically changed to "utf-8".
	// 
	void put_CommandCharset(const char *newVal);

	// If the Connect method fails, this property can be checked to determine the
	// reason for failure.
	// 
	// Possible values are:
	// 0 = success
	// 
	// Normal (non-TLS) sockets:
	// 1 = empty hostname
	// 2 = DNS lookup failed
	// 3 = DNS timeout
	// 4 = Aborted by application.
	// 5 = Internal failure.
	// 6 = Connect Timed Out
	// 7 = Connect Rejected (or failed for some other reason)
	// 
	// SSL/TLS:
	// 100 = TLS internal error.
	// 101 = Failed to send client hello.
	// 102 = Unexpected handshake message.
	// 103 = Failed to read server hello.
	// 104 = No server certificate.
	// 105 = Unexpected TLS protocol version.
	// 106 = Server certificate verify failed (the server certificate is expired or the cert's signature verification failed).
	// 107 = Unacceptable TLS protocol version.
	// 109 = Failed to read handshake messages.
	// 110 = Failed to send client certificate handshake message.
	// 111 = Failed to send client key exchange handshake message.
	// 112 = Client certificate's private key not accessible.
	// 113 = Failed to send client cert verify handshake message.
	// 114 = Failed to send change cipher spec handshake message.
	// 115 = Failed to send finished handshake message.
	// 116 = Server's Finished message is invalid.
	// 
	// FTP:
	// 200 = Connected, but failed to receive greeting from FTP server.
	// 201 = Failed to do AUTH TLS or AUTH SSL.
	// Protocol/Component:
	// 300 = asynch op in progress
	// 301 = login failure.
	// 
	int get_ConnectFailReason(void);

	// Maximum number of seconds to wait when connecting to an FTP server. The default
	// is 30 seconds. A value of 0 indicates the willingness to wait forever.
	int get_ConnectTimeout(void);
	// Maximum number of seconds to wait when connecting to an FTP server. The default
	// is 30 seconds. A value of 0 indicates the willingness to wait forever.
	void put_ConnectTimeout(int newVal);

	// True if the FTP2 component was able to establish a TCP/IP connection to the FTP
	// server after calling Connect.
	bool get_ConnectVerified(void);

	// Used to control CRLF line endings when downloading text files in ASCII mode. The
	// default value is 0.
	// 
	// Possible values are:
	// 0 = Do nothing.  The line-endings are not modified as received from the FTP server.
	// 1 = Convert all line-endings to CR+LF
	// 2 = Convert all line-endings to bare LF's
	// 3 = Convert all line-endings to bare CR's
	// 
	int get_CrlfMode(void);
	// Used to control CRLF line endings when downloading text files in ASCII mode. The
	// default value is 0.
	// 
	// Possible values are:
	// 0 = Do nothing.  The line-endings are not modified as received from the FTP server.
	// 1 = Convert all line-endings to CR+LF
	// 2 = Convert all line-endings to bare LF's
	// 3 = Convert all line-endings to bare CR's
	// 
	void put_CrlfMode(int newVal);

	// Controls the data protection level for the data connections. Possible values are
	// "control", "clear", or "private".
	//     "control" is the default, and the data connections will be the same as for
	//     the control connection. If the control connection is SSL/TLS, then the data
	//     connections are also SSL/TLS. If the control connection is unencrypted, then the
	//     data connections will also be unencrypted.
	//     "clear" means that the data connections will always be unencrypted (TCP
	//     without TLS).
	//     "private" means that the data connections will always be encrypted (TLS).
	void get_DataProtection(CkString &str);
	// Controls the data protection level for the data connections. Possible values are
	// "control", "clear", or "private".
	//     "control" is the default, and the data connections will be the same as for
	//     the control connection. If the control connection is SSL/TLS, then the data
	//     connections are also SSL/TLS. If the control connection is unencrypted, then the
	//     data connections will also be unencrypted.
	//     "clear" means that the data connections will always be unencrypted (TCP
	//     without TLS).
	//     "private" means that the data connections will always be encrypted (TLS).
	const char *dataProtection(void);
	// Controls the data protection level for the data connections. Possible values are
	// "control", "clear", or "private".
	//     "control" is the default, and the data connections will be the same as for
	//     the control connection. If the control connection is SSL/TLS, then the data
	//     connections are also SSL/TLS. If the control connection is unencrypted, then the
	//     data connections will also be unencrypted.
	//     "clear" means that the data connections will always be unencrypted (TCP
	//     without TLS).
	//     "private" means that the data connections will always be encrypted (TLS).
	void put_DataProtection(const char *newVal);

	// Indicates the charset of the directory listings received from the FTP server.
	// The FTP2 client must interpret the directory listing bytes using the correct
	// character encoding in order to correctly receive non-English characters. The
	// default value of this property is "ansi".
	// 
	// This property may be updated to "utf-8" after connecting because a "FEAT"
	// command is automatically sent to get the features of the FTP server. If UTF8 is
	// indicated as a feature, then this property is automatically changed to "utf-8".
	// 
	void get_DirListingCharset(CkString &str);
	// Indicates the charset of the directory listings received from the FTP server.
	// The FTP2 client must interpret the directory listing bytes using the correct
	// character encoding in order to correctly receive non-English characters. The
	// default value of this property is "ansi".
	// 
	// This property may be updated to "utf-8" after connecting because a "FEAT"
	// command is automatically sent to get the features of the FTP server. If UTF8 is
	// indicated as a feature, then this property is automatically changed to "utf-8".
	// 
	const char *dirListingCharset(void);
	// Indicates the charset of the directory listings received from the FTP server.
	// The FTP2 client must interpret the directory listing bytes using the correct
	// character encoding in order to correctly receive non-English characters. The
	// default value of this property is "ansi".
	// 
	// This property may be updated to "utf-8" after connecting because a "FEAT"
	// command is automatically sent to get the features of the FTP server. If UTF8 is
	// indicated as a feature, then this property is automatically changed to "utf-8".
	// 
	void put_DirListingCharset(const char *newVal);

	// The average download rate in bytes/second. This property is updated in real-time
	// during any FTP download (asynchronous or synchronous).
	int get_DownloadTransferRate(void);

	// If set, forces the IP address used in the PORT command for Active mode (i.e.
	// non-passive) data transfers. This string property should be set to the IP
	// address in dotted notation, such as "233.190.65.31".
	// 
	// Note: This property can also be set to the special keyword "control" to force
	// the PORT IP address to be the address of the control connection's peer.
	// 
	// Starting in v9.5.0.58, the IP address can be prefixed with the string "bind-".
	// For example, "bind-233.190.65.31". When "bind-" is specified, the local data
	// socket will be bound to the IP address when created. Otherwise, the IP address
	// is only used as the argument to the PORT command that is sent to the server.
	// 
	void get_ForcePortIpAddress(CkString &str);
	// If set, forces the IP address used in the PORT command for Active mode (i.e.
	// non-passive) data transfers. This string property should be set to the IP
	// address in dotted notation, such as "233.190.65.31".
	// 
	// Note: This property can also be set to the special keyword "control" to force
	// the PORT IP address to be the address of the control connection's peer.
	// 
	// Starting in v9.5.0.58, the IP address can be prefixed with the string "bind-".
	// For example, "bind-233.190.65.31". When "bind-" is specified, the local data
	// socket will be bound to the IP address when created. Otherwise, the IP address
	// is only used as the argument to the PORT command that is sent to the server.
	// 
	const char *forcePortIpAddress(void);
	// If set, forces the IP address used in the PORT command for Active mode (i.e.
	// non-passive) data transfers. This string property should be set to the IP
	// address in dotted notation, such as "233.190.65.31".
	// 
	// Note: This property can also be set to the special keyword "control" to force
	// the PORT IP address to be the address of the control connection's peer.
	// 
	// Starting in v9.5.0.58, the IP address can be prefixed with the string "bind-".
	// For example, "bind-233.190.65.31". When "bind-" is specified, the local data
	// socket will be bound to the IP address when created. Otherwise, the IP address
	// is only used as the argument to the PORT command that is sent to the server.
	// 
	void put_ForcePortIpAddress(const char *newVal);

	// The initial greeting received from the FTP server after connecting.
	void get_Greeting(CkString &str);
	// The initial greeting received from the FTP server after connecting.
	const char *greeting(void);

	// Chilkat FTP2 supports MODE Z, which is a transfer mode implemented by some FTP
	// servers. It allows for files to be uploaded and downloaded using compressed
	// streams (using the zlib deflate algorithm). This is a read-only property. It
	// will be set to true if the FTP2 component detects that your FTP server
	// supports MODE Z. Otherwise it is set to false.
	bool get_HasModeZ(void);

	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any FTP operation prior to
	// completion. If HeartbeatMs is 0, no AbortCheck event callbacks will occur. Also,
	// AbortCheck callbacks do not occur when doing asynchronous transfers.
	int get_HeartbeatMs(void);
	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any FTP operation prior to
	// completion. If HeartbeatMs is 0, no AbortCheck event callbacks will occur. Also,
	// AbortCheck callbacks do not occur when doing asynchronous transfers.
	void put_HeartbeatMs(int newVal);

	// The domain name of the FTP server. May also use the IPv4 or IPv6 address in
	// string format.
	void get_Hostname(CkString &str);
	// The domain name of the FTP server. May also use the IPv4 or IPv6 address in
	// string format.
	const char *hostname(void);
	// The domain name of the FTP server. May also use the IPv4 or IPv6 address in
	// string format.
	void put_Hostname(const char *newVal);

	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	void get_HttpProxyAuthMethod(CkString &str);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	const char *httpProxyAuthMethod(void);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	void put_HttpProxyAuthMethod(const char *newVal);

	// If an HTTP proxy is used, and it uses NTLM authentication, then this optional
	// property is the NTLM authentication domain.
	void get_HttpProxyDomain(CkString &str);
	// If an HTTP proxy is used, and it uses NTLM authentication, then this optional
	// property is the NTLM authentication domain.
	const char *httpProxyDomain(void);
	// If an HTTP proxy is used, and it uses NTLM authentication, then this optional
	// property is the NTLM authentication domain.
	void put_HttpProxyDomain(const char *newVal);

	// If an HTTP proxy is to be used, set this property to the HTTP proxy hostname or
	// IPv4 address (in dotted decimal notation).
	void get_HttpProxyHostname(CkString &str);
	// If an HTTP proxy is to be used, set this property to the HTTP proxy hostname or
	// IPv4 address (in dotted decimal notation).
	const char *httpProxyHostname(void);
	// If an HTTP proxy is to be used, set this property to the HTTP proxy hostname or
	// IPv4 address (in dotted decimal notation).
	void put_HttpProxyHostname(const char *newVal);

	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy password.
	void get_HttpProxyPassword(CkString &str);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy password.
	const char *httpProxyPassword(void);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy password.
	void put_HttpProxyPassword(const char *newVal);

	// If an HTTP proxy is to be used, set this property to the HTTP proxy port number.
	// (Two commonly used HTTP proxy ports are 8080 and 3128.)
	int get_HttpProxyPort(void);
	// If an HTTP proxy is to be used, set this property to the HTTP proxy port number.
	// (Two commonly used HTTP proxy ports are 8080 and 3128.)
	void put_HttpProxyPort(int newVal);

	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy login name.
	void get_HttpProxyUsername(CkString &str);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy login name.
	const char *httpProxyUsername(void);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy login name.
	void put_HttpProxyUsername(const char *newVal);

	// Forces a timeout when a response is expected on the control channel, but no
	// response arrives for this number of milliseconds. Setting IdleTimeoutMs = 0
	// allows the application to wait indefinitely. The default value is 60000 (i.e. 60
	// seconds).
	int get_IdleTimeoutMs(void);
	// Forces a timeout when a response is expected on the control channel, but no
	// response arrives for this number of milliseconds. Setting IdleTimeoutMs = 0
	// allows the application to wait indefinitely. The default value is 60000 (i.e. 60
	// seconds).
	void put_IdleTimeoutMs(int newVal);

	// Important: This property is deprecated. Applications should instead call the
	// CheckConnection method.
	// 
	// Returns true if currently connected and logged into an FTP server, otherwise
	// returns false.
	// 
	// Note: Accessing this property may cause a NOOP command to be sent to the FTP
	// server.
	// 
	bool get_IsConnected(void);

	// Turns the in-memory session logging on or off. If on, the session log can be
	// obtained via the SessionLog property.
	bool get_KeepSessionLog(void);
	// Turns the in-memory session logging on or off. If on, the session log can be
	// obtained via the SessionLog property.
	void put_KeepSessionLog(bool newVal);

	// Enables internal features that can help when uploading or downloading extremely
	// large files. In some cases, if the time required to transfer a file is long, the
	// control connection is closed by the server or other network infrastructure
	// because it was idle for so long. Setting this property equal to true will keep
	// the control connection very slightly used to prevent this from happening.
	// 
	// The default value of this property is false. This property should only be set
	// to true if this sort of problem is encountered.
	// 
	bool get_LargeFileMeasures(void);
	// Enables internal features that can help when uploading or downloading extremely
	// large files. In some cases, if the time required to transfer a file is long, the
	// control connection is closed by the server or other network infrastructure
	// because it was idle for so long. Setting this property equal to true will keep
	// the control connection very slightly used to prevent this from happening.
	// 
	// The default value of this property is false. This property should only be set
	// to true if this sort of problem is encountered.
	// 
	void put_LargeFileMeasures(bool newVal);

	// Contains the last control-channel reply. For example: "550 Failed to change
	// directory." or "250 Directory successfully changed." The control channel reply
	// is typically formatted as an integer status code followed by a one-line
	// description.
	void get_LastReply(CkString &str);
	// Contains the last control-channel reply. For example: "550 Failed to change
	// directory." or "250 Directory successfully changed." The control channel reply
	// is typically formatted as an integer status code followed by a one-line
	// description.
	const char *lastReply(void);

	// A wildcard pattern, defaulting to "*" that determines the files and directories
	// included in the following properties and methods: GetDirCount, GetCreateTime,
	// GetFilename, GetIsDirectory, GetLastAccessTime, GetModifiedTime, GetSize.
	// 
	// Note: Do not include a directory path in the ListPattern. For example, do not
	// set the ListPattern equal to a string such as this: "subdir/*.txt". The correct
	// solution is to first change the remote directory to "subdir" by calling
	// ChangeRemoteDir, and then set the ListPattern equal to "*.txt".
	// 
	void get_ListPattern(CkString &str);
	// A wildcard pattern, defaulting to "*" that determines the files and directories
	// included in the following properties and methods: GetDirCount, GetCreateTime,
	// GetFilename, GetIsDirectory, GetLastAccessTime, GetModifiedTime, GetSize.
	// 
	// Note: Do not include a directory path in the ListPattern. For example, do not
	// set the ListPattern equal to a string such as this: "subdir/*.txt". The correct
	// solution is to first change the remote directory to "subdir" by calling
	// ChangeRemoteDir, and then set the ListPattern equal to "*.txt".
	// 
	const char *listPattern(void);
	// A wildcard pattern, defaulting to "*" that determines the files and directories
	// included in the following properties and methods: GetDirCount, GetCreateTime,
	// GetFilename, GetIsDirectory, GetLastAccessTime, GetModifiedTime, GetSize.
	// 
	// Note: Do not include a directory path in the ListPattern. For example, do not
	// set the ListPattern equal to a string such as this: "subdir/*.txt". The correct
	// solution is to first change the remote directory to "subdir" by calling
	// ChangeRemoteDir, and then set the ListPattern equal to "*.txt".
	// 
	void put_ListPattern(const char *newVal);

	// True if the FTP2 component was able to login to the FTP server after calling
	// Connect.
	bool get_LoginVerified(void);

	// Important: This property is deprecated. Applications should instead call the
	// GetDirCount method.
	// 
	// The number of files and sub-directories in the current remote directory that
	// match the ListPattern. (The ListPattern defaults to "*", so unless changed, this
	// is the total number of files and sub-directories.)
	// 
	// Important: Accessing this property can cause the directory listing to be
	// retrieved from the FTP server. For FTP servers that doe not support the
	// MLST/MLSD commands, this is technically a data transfer that requires a
	// temporary data connection to be established in the same way as when uploading or
	// downloading files. If your program hangs while accessing NumFilesAndDirs, it
	// probably means that the data connection could not be established. The most
	// common solution is to switch to using Passive mode by setting the Passive
	// property = true. If this does not help, examine the contents of the
	// LastErrorText property after NumFilesAndDirs finally returns (after timing out).
	// Also, seethis Chilkat blog post about FTP connection settings
	// <http://www.cknotes.com/?p=282>.
	// 
	int get_NumFilesAndDirs(void);

	// A read-only property that indicates whether a partial transfer was received in
	// the last method call to download a file. Set to true if a partial transfer was
	// received. Set to false if nothing was received, or if the full file was
	// received.
	bool get_PartialTransfer(void);

	// Set to true for FTP to operate in passive mode, otherwise set to false for
	// non-passive (.i.e. "active" or "port" mode). The default value of this property
	// is true.
	bool get_Passive(void);
	// Set to true for FTP to operate in passive mode, otherwise set to false for
	// non-passive (.i.e. "active" or "port" mode). The default value of this property
	// is true.
	void put_Passive(bool newVal);

	// This can handle problems that may arise when an FTP server is located behind a
	// NAT router. FTP servers respond to the PASV command by sending the IP address
	// and port where it will be listening for the data connection. If the control
	// connection is SSL encrypted, the NAT router is not able to convert from an
	// internal IP address (typically beginning with 192.168) to an external address.
	// When set to true, PassiveUseHostAddr property tells the FTP client to discard
	// the IP address part of the PASV response and replace it with the IP address of
	// the already-established control connection. The default value of this property
	// is true.
	bool get_PassiveUseHostAddr(void);
	// This can handle problems that may arise when an FTP server is located behind a
	// NAT router. FTP servers respond to the PASV command by sending the IP address
	// and port where it will be listening for the data connection. If the control
	// connection is SSL encrypted, the NAT router is not able to convert from an
	// internal IP address (typically beginning with 192.168) to an external address.
	// When set to true, PassiveUseHostAddr property tells the FTP client to discard
	// the IP address part of the PASV response and replace it with the IP address of
	// the already-established control connection. The default value of this property
	// is true.
	void put_PassiveUseHostAddr(bool newVal);

	// Password for logging into the FTP server.
	void get_Password(CkString &str);
	// Password for logging into the FTP server.
	const char *password(void);
	// Password for logging into the FTP server.
	void put_Password(const char *newVal);

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

	// Port number. Automatically defaults to the default port for the FTP service.
	int get_Port(void);
	// Port number. Automatically defaults to the default port for the FTP service.
	void put_Port(int newVal);

	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	bool get_PreferIpv6(void);
	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	void put_PreferIpv6(bool newVal);

	// If true, the NLST command is used instead of LIST when fetching a directory
	// listing. This can help in very rare cases where the FTP server returns truncated
	// filenames. The drawback to using NLST is that it won't return size or date/time
	// info (but it should return the full filename).
	// 
	// The default value of this property is false.
	// 
	bool get_PreferNlst(void);
	// If true, the NLST command is used instead of LIST when fetching a directory
	// listing. This can help in very rare cases where the FTP server returns truncated
	// filenames. The drawback to using NLST is that it won't return size or date/time
	// info (but it should return the full filename).
	// 
	// The default value of this property is false.
	// 
	void put_PreferNlst(bool newVal);

	// Progress monitoring for FTP downloads rely on the FTP server indicating the file
	// size within the RETR response. Some FTP servers however, do not indicate the
	// file size and therefore it is not possible to monitor progress based on
	// percentage completion. This property allows the application to explicitly tell
	// the FTP component the size of the file about to be downloaded for the next
	// GetFile call.
	int get_ProgressMonSize(void);
	// Progress monitoring for FTP downloads rely on the FTP server indicating the file
	// size within the RETR response. Some FTP servers however, do not indicate the
	// file size and therefore it is not possible to monitor progress based on
	// percentage completion. This property allows the application to explicitly tell
	// the FTP component the size of the file about to be downloaded for the next
	// GetFile call.
	void put_ProgressMonSize(int newVal);

	// Same as ProgressMonSize, but allows for sizes greater than the 32-bit integer
	// limit.
	__int64 get_ProgressMonSize64(void);
	// Same as ProgressMonSize, but allows for sizes greater than the 32-bit integer
	// limit.
	void put_ProgressMonSize64(__int64 newVal);

	// The hostname of your FTP proxy, if a proxy server is used.
	void get_ProxyHostname(CkString &str);
	// The hostname of your FTP proxy, if a proxy server is used.
	const char *proxyHostname(void);
	// The hostname of your FTP proxy, if a proxy server is used.
	void put_ProxyHostname(const char *newVal);

	// The proxy scheme used by your FTP proxy server. Valid values are 0 to 9. The
	// default value is 0 which indicates that no proxy server is used. Supported proxy
	// methods are as follows:
	// 
	// Note: The ProxyHostname is the hostname of the firewall, if the proxy is a
	// firewall. Also, the ProxyUsername and ProxyPassword are the firewall
	// username/password (if the proxy is a firewall).
	// 
	//     ProxyMethod = 1 (SITE site)
	// 
	//         USER ProxyUsername
	//         PASS ProxyPassword
	//         SITE Hostname
	//         USER Username
	//         PASS Password
	// 
	//     ProxyMethod = 2 (USER user@site)
	// 
	//         USER Username@Hostname:Port
	//         PASS Password
	// 
	//     ProxyMethod = 3 (USER with login)
	// 
	//         USER ProxyUsername
	//         PASS ProxyPassword
	//         USER Username@Hostname:Port
	//         PASS Password
	// 
	//     ProxyMethod = 4 (USER/PASS/ACCT)
	// 
	//         USER Username@Hostname:Port ProxyUsername
	//         PASS Password
	//         ACCT ProxyPassword
	// 
	//     ProxyMethod = 5 (OPEN site)
	// 
	//         USER ProxyUsername
	//         PASS ProxyPassword
	//         OPEN Hostname
	//         USER Username
	//         PASS Password
	// 
	//     ProxyMethod = 6 (firewallId@site)
	// 
	//         USER ProxyUsername@Hostname
	//         USER Username
	//         PASS Password
	// 
	//     ProxyMethod = 7
	// 
	//         USER ProxyUsername
	//         USER ProxyPassword
	//         SITE Hostname:Port USER Username
	//         PASS Password
	// 
	//     ProxyMethod = 8
	// 
	//         USER Username@ProxyUsername@Hostname
	//         PASS Password@ProxyPassword
	// 
	//     ProxyMethod = 9
	// 
	//         ProxyUsername ProxyPassword Username Password
	// 
	int get_ProxyMethod(void);
	// The proxy scheme used by your FTP proxy server. Valid values are 0 to 9. The
	// default value is 0 which indicates that no proxy server is used. Supported proxy
	// methods are as follows:
	// 
	// Note: The ProxyHostname is the hostname of the firewall, if the proxy is a
	// firewall. Also, the ProxyUsername and ProxyPassword are the firewall
	// username/password (if the proxy is a firewall).
	// 
	//     ProxyMethod = 1 (SITE site)
	// 
	//         USER ProxyUsername
	//         PASS ProxyPassword
	//         SITE Hostname
	//         USER Username
	//         PASS Password
	// 
	//     ProxyMethod = 2 (USER user@site)
	// 
	//         USER Username@Hostname:Port
	//         PASS Password
	// 
	//     ProxyMethod = 3 (USER with login)
	// 
	//         USER ProxyUsername
	//         PASS ProxyPassword
	//         USER Username@Hostname:Port
	//         PASS Password
	// 
	//     ProxyMethod = 4 (USER/PASS/ACCT)
	// 
	//         USER Username@Hostname:Port ProxyUsername
	//         PASS Password
	//         ACCT ProxyPassword
	// 
	//     ProxyMethod = 5 (OPEN site)
	// 
	//         USER ProxyUsername
	//         PASS ProxyPassword
	//         OPEN Hostname
	//         USER Username
	//         PASS Password
	// 
	//     ProxyMethod = 6 (firewallId@site)
	// 
	//         USER ProxyUsername@Hostname
	//         USER Username
	//         PASS Password
	// 
	//     ProxyMethod = 7
	// 
	//         USER ProxyUsername
	//         USER ProxyPassword
	//         SITE Hostname:Port USER Username
	//         PASS Password
	// 
	//     ProxyMethod = 8
	// 
	//         USER Username@ProxyUsername@Hostname
	//         PASS Password@ProxyPassword
	// 
	//     ProxyMethod = 9
	// 
	//         ProxyUsername ProxyPassword Username Password
	// 
	void put_ProxyMethod(int newVal);

	// The password for authenticating with the FTP proxy server.
	void get_ProxyPassword(CkString &str);
	// The password for authenticating with the FTP proxy server.
	const char *proxyPassword(void);
	// The password for authenticating with the FTP proxy server.
	void put_ProxyPassword(const char *newVal);

	// If an FTP proxy server is used, this is the port number at which the proxy
	// server is listening for connections.
	int get_ProxyPort(void);
	// If an FTP proxy server is used, this is the port number at which the proxy
	// server is listening for connections.
	void put_ProxyPort(int newVal);

	// The username for authenticating with the FTP proxy server.
	void get_ProxyUsername(CkString &str);
	// The username for authenticating with the FTP proxy server.
	const char *proxyUsername(void);
	// The username for authenticating with the FTP proxy server.
	void put_ProxyUsername(const char *newVal);

	// Forces a timeout when incoming data is expected on a data channel, but no data
	// arrives for this number of seconds. The ReadTimeout is the amount of time that
	// needs to elapse while no additional data is forthcoming. During a long download,
	// if the data stream halts for more than this amount, it will timeout. Otherwise,
	// there is no limit on the length of time for the entire download.
	// 
	// The default value is 60.
	// 
	int get_ReadTimeout(void);
	// Forces a timeout when incoming data is expected on a data channel, but no data
	// arrives for this number of seconds. The ReadTimeout is the amount of time that
	// needs to elapse while no additional data is forthcoming. During a long download,
	// if the data stream halts for more than this amount, it will timeout. Otherwise,
	// there is no limit on the length of time for the entire download.
	// 
	// The default value is 60.
	// 
	void put_ReadTimeout(int newVal);

	// If true, then the FTP2 client will verify the server's SSL certificate. The
	// server's certificate signature is verified with its issuer, and the issuer's
	// cert is verified with its issuer, etc. up to the root CA cert. If a signature
	// verification fails, the connection is not allowed. Also, if the certificate is
	// expired, or if the cert's signature is invalid, the connection is not allowed.
	// The default value of this property is false.
	bool get_RequireSslCertVerify(void);
	// If true, then the FTP2 client will verify the server's SSL certificate. The
	// server's certificate signature is verified with its issuer, and the issuer's
	// cert is verified with its issuer, etc. up to the root CA cert. If a signature
	// verification fails, the connection is not allowed. Also, if the certificate is
	// expired, or if the cert's signature is invalid, the connection is not allowed.
	// The default value of this property is false.
	void put_RequireSslCertVerify(bool newVal);

	// Both uploads and downloads may be resumed by simply setting this property =
	// true and re-calling the upload or download method.
	bool get_RestartNext(void);
	// Both uploads and downloads may be resumed by simply setting this property =
	// true and re-calling the upload or download method.
	void put_RestartNext(bool newVal);

	// Contains the session log if KeepSessionLog is turned on.
	void get_SessionLog(CkString &str);
	// Contains the session log if KeepSessionLog is turned on.
	const char *sessionLog(void);

	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	void get_SocksHostname(CkString &str);
	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	const char *socksHostname(void);
	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	void put_SocksHostname(const char *newVal);

	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	void get_SocksPassword(CkString &str);
	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	const char *socksPassword(void);
	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	void put_SocksPassword(const char *newVal);

	// The SOCKS4/SOCKS5 proxy port. The default value is 1080. This property only
	// applies if a SOCKS proxy is used (if the SocksVersion property is set to 4 or
	// 5).
	int get_SocksPort(void);
	// The SOCKS4/SOCKS5 proxy port. The default value is 1080. This property only
	// applies if a SOCKS proxy is used (if the SocksVersion property is set to 4 or
	// 5).
	void put_SocksPort(int newVal);

	// The SOCKS4/SOCKS5 proxy username. This property is only used if the SocksVersion
	// property is set to 4 or 5).
	void get_SocksUsername(CkString &str);
	// The SOCKS4/SOCKS5 proxy username. This property is only used if the SocksVersion
	// property is set to 4 or 5).
	const char *socksUsername(void);
	// The SOCKS4/SOCKS5 proxy username. This property is only used if the SocksVersion
	// property is set to 4 or 5).
	void put_SocksUsername(const char *newVal);

	// SocksVersion May be set to one of the following integer values:
	// 
	// 0 - No SOCKS proxy is used. This is the default.
	// 4 - Connect via a SOCKS4 proxy.
	// 5 - Connect via a SOCKS5 proxy.
	// 
	int get_SocksVersion(void);
	// SocksVersion May be set to one of the following integer values:
	// 
	// 0 - No SOCKS proxy is used. This is the default.
	// 4 - Connect via a SOCKS4 proxy.
	// 5 - Connect via a SOCKS5 proxy.
	// 
	void put_SocksVersion(int newVal);

	// Sets the receive buffer size socket option. Normally, this property should be
	// left unchanged. The default value is 4194304.
	// 
	// This property can be increased if download performance seems slow. It is
	// recommended to be a multiple of 4096.
	// 
	// Note: This property only applies to FTP data connections. The FTP control
	// connection is not used for uploading or downloading files, and is therefore not
	// performance sensitive.
	// 
	int get_SoRcvBuf(void);
	// Sets the receive buffer size socket option. Normally, this property should be
	// left unchanged. The default value is 4194304.
	// 
	// This property can be increased if download performance seems slow. It is
	// recommended to be a multiple of 4096.
	// 
	// Note: This property only applies to FTP data connections. The FTP control
	// connection is not used for uploading or downloading files, and is therefore not
	// performance sensitive.
	// 
	void put_SoRcvBuf(int newVal);

	// Sets the send buffer size socket option. Normally, this property should be left
	// unchanged. The default value is 262144.
	// 
	// This property can be increased if upload performance seems slow. It is
	// recommended to be a multiple of 4096. Testing with sizes such as 512K and 1MB is
	// reasonable.
	// 
	// Note: This property only applies to FTP data connections. The FTP control
	// connection is not used for uploading or downloading files, and is therefore not
	// performance sensitive.
	// 
	int get_SoSndBuf(void);
	// Sets the send buffer size socket option. Normally, this property should be left
	// unchanged. The default value is 262144.
	// 
	// This property can be increased if upload performance seems slow. It is
	// recommended to be a multiple of 4096. Testing with sizes such as 512K and 1MB is
	// reasonable.
	// 
	// Note: This property only applies to FTP data connections. The FTP control
	// connection is not used for uploading or downloading files, and is therefore not
	// performance sensitive.
	// 
	void put_SoSndBuf(int newVal);

	// Use TLS/SSL for FTP connections. You would typically set Ssl = true when
	// connecting to port 990 on FTP servers that support TLS/SSL mode. Note: It is
	// more common to use AuthTls.
	bool get_Ssl(void);
	// Use TLS/SSL for FTP connections. You would typically set Ssl = true when
	// connecting to port 990 on FTP servers that support TLS/SSL mode. Note: It is
	// more common to use AuthTls.
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

	// Selects the secure protocol to be used for secure (SSL/TLS) implicit and
	// explicit (AUTH TLS / AUTH SSL) connections . Possible values are:
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
	// Selects the secure protocol to be used for secure (SSL/TLS) implicit and
	// explicit (AUTH TLS / AUTH SSL) connections . Possible values are:
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
	// Selects the secure protocol to be used for secure (SSL/TLS) implicit and
	// explicit (AUTH TLS / AUTH SSL) connections . Possible values are:
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

	// Read-only property that returns true if the FTP server's digital certificate
	// was verified when connecting via SSL / TLS.
	bool get_SslServerCertVerified(void);

	// If true, then empty directories on the server are created locally when doing a
	// download synchronization. If false, then only directories containing files
	// that are downloaded are auto-created.
	// 
	// The default value of this property is true.
	// 
	bool get_SyncCreateAllLocalDirs(void);
	// If true, then empty directories on the server are created locally when doing a
	// download synchronization. If false, then only directories containing files
	// that are downloaded are auto-created.
	// 
	// The default value of this property is true.
	// 
	void put_SyncCreateAllLocalDirs(bool newVal);

	// The paths of the files uploaded or downloaded in the last call to
	// SyncDeleteTree, SyncLocalDir, SyncLocalTree, SyncRemoteTree, or SyncRemoteTree2.
	// The paths are listed one per line. In both cases (for upload and download) each
	// line contains the paths relative to the root synced directory.
	void get_SyncedFiles(CkString &str);
	// The paths of the files uploaded or downloaded in the last call to
	// SyncDeleteTree, SyncLocalDir, SyncLocalTree, SyncRemoteTree, or SyncRemoteTree2.
	// The paths are listed one per line. In both cases (for upload and download) each
	// line contains the paths relative to the root synced directory.
	const char *syncedFiles(void);
	// The paths of the files uploaded or downloaded in the last call to
	// SyncDeleteTree, SyncLocalDir, SyncLocalTree, SyncRemoteTree, or SyncRemoteTree2.
	// The paths are listed one per line. In both cases (for upload and download) each
	// line contains the paths relative to the root synced directory.
	void put_SyncedFiles(const char *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the Sync* upload and download methods
	// will only transfer files that match any one of these patterns. Pattern matching
	// is case-insensitive.
	// 
	// Note: Starting in version 9.5.0.47, this property also applies to the
	// DownloadTree and DirTreeXml methods.
	// 
	void get_SyncMustMatch(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the Sync* upload and download methods
	// will only transfer files that match any one of these patterns. Pattern matching
	// is case-insensitive.
	// 
	// Note: Starting in version 9.5.0.47, this property also applies to the
	// DownloadTree and DirTreeXml methods.
	// 
	const char *syncMustMatch(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the Sync* upload and download methods
	// will only transfer files that match any one of these patterns. Pattern matching
	// is case-insensitive.
	// 
	// Note: Starting in version 9.5.0.47, this property also applies to the
	// DownloadTree and DirTreeXml methods.
	// 
	void put_SyncMustMatch(const char *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the Sync* upload and download methods will
	// only enter directories that match any one of these patterns. Pattern matching is
	// case-insensitive.
	void get_SyncMustMatchDir(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the Sync* upload and download methods will
	// only enter directories that match any one of these patterns. Pattern matching is
	// case-insensitive.
	const char *syncMustMatchDir(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the Sync* upload and download methods will
	// only enter directories that match any one of these patterns. Pattern matching is
	// case-insensitive.
	void put_SyncMustMatchDir(const char *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the Sync* upload and download methods
	// will not transfer files that match any one of these patterns. Pattern matching
	// is case-insensitive.
	// 
	// Note: Starting in version 9.5.0.47, this property also applies to the
	// DownloadTree and DirTreeXml methods.
	// 
	void get_SyncMustNotMatch(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the Sync* upload and download methods
	// will not transfer files that match any one of these patterns. Pattern matching
	// is case-insensitive.
	// 
	// Note: Starting in version 9.5.0.47, this property also applies to the
	// DownloadTree and DirTreeXml methods.
	// 
	const char *syncMustNotMatch(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the Sync* upload and download methods
	// will not transfer files that match any one of these patterns. Pattern matching
	// is case-insensitive.
	// 
	// Note: Starting in version 9.5.0.47, this property also applies to the
	// DownloadTree and DirTreeXml methods.
	// 
	void put_SyncMustNotMatch(const char *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the Sync* upload and download methods will
	// enter directories that match any one of these patterns. Pattern matching is
	// case-insensitive.
	void get_SyncMustNotMatchDir(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the Sync* upload and download methods will
	// enter directories that match any one of these patterns. Pattern matching is
	// case-insensitive.
	const char *syncMustNotMatchDir(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the Sync* upload and download methods will
	// enter directories that match any one of these patterns. Pattern matching is
	// case-insensitive.
	void put_SyncMustNotMatchDir(const char *newVal);

	// Contains the list of files that would be transferred in a call to
	// SyncRemoteTree2 when the previewOnly argument is set to true. This string
	// property contains one filepath per line, separated by CRLF line endings. After
	// SyncRemoteTree2 is called, this property contains the filepaths of the local
	// files that would be uploaded to the FTP server.
	void get_SyncPreview(CkString &str);
	// Contains the list of files that would be transferred in a call to
	// SyncRemoteTree2 when the previewOnly argument is set to true. This string
	// property contains one filepath per line, separated by CRLF line endings. After
	// SyncRemoteTree2 is called, this property contains the filepaths of the local
	// files that would be uploaded to the FTP server.
	const char *syncPreview(void);

	// Contains the current or last negotiated TLS cipher suite. If no TLS connection
	// has yet to be established, or if a connection as attempted and failed, then this
	// will be empty. A sample cipher suite string looks like this:
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256.
	void get_TlsCipherSuite(CkString &str);
	// Contains the current or last negotiated TLS cipher suite. If no TLS connection
	// has yet to be established, or if a connection as attempted and failed, then this
	// will be empty. A sample cipher suite string looks like this:
	// TLS_DHE_RSA_WITH_AES_256_CBC_SHA256.
	const char *tlsCipherSuite(void);

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

	// Contains the current or last negotiated TLS protocol version. If no TLS
	// connection has yet to be established, or if a connection as attempted and
	// failed, then this will be empty. Possible values are "SSL 3.0", "TLS 1.0", "TLS
	// 1.1", "TLS 1.2", and "TLS 1.3".
	void get_TlsVersion(CkString &str);
	// Contains the current or last negotiated TLS protocol version. If no TLS
	// connection has yet to be established, or if a connection as attempted and
	// failed, then this will be empty. Possible values are "SSL 3.0", "TLS 1.0", "TLS
	// 1.1", "TLS 1.2", and "TLS 1.3".
	const char *tlsVersion(void);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "OpenNonExclusive" - Introduced in v9.5.0.78. When downloading files on
	//     Windows systems, open the local file with non-exclusive access to allow other
	//     programs the ability to access the file as it's being downloaded.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "EnableTls13" - Introduced in v9.5.0.82. Causes TLS 1.3 to be offered in the
	//     ClientHello of the TLS protocol, allowing the server to select TLS 1.3 for the
	//     session. Future versions of Chilkat will enable TLS 1.3 by default. This option
	//     is only necessary in v9.5.0.82 if TLS 1.3 is desired.
	//     "DisableTls13" - Disables the use of TLS 1.3. TLS 1.3 is enabled by default
	//     in Chilkat v9.5.0.84 and above. This keyword can be used to avoid TLS 1.3 if it
	//     causes problems.
	//     "NoPreserveFileTime" - Introduced in v9.5.0.85. Downloaded files will get
	//     the current local system date/time and no attempt will be made to try to
	//     preserver the last-modified date/time of the file on the server.
	// 
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "OpenNonExclusive" - Introduced in v9.5.0.78. When downloading files on
	//     Windows systems, open the local file with non-exclusive access to allow other
	//     programs the ability to access the file as it's being downloaded.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "EnableTls13" - Introduced in v9.5.0.82. Causes TLS 1.3 to be offered in the
	//     ClientHello of the TLS protocol, allowing the server to select TLS 1.3 for the
	//     session. Future versions of Chilkat will enable TLS 1.3 by default. This option
	//     is only necessary in v9.5.0.82 if TLS 1.3 is desired.
	//     "DisableTls13" - Disables the use of TLS 1.3. TLS 1.3 is enabled by default
	//     in Chilkat v9.5.0.84 and above. This keyword can be used to avoid TLS 1.3 if it
	//     causes problems.
	//     "NoPreserveFileTime" - Introduced in v9.5.0.85. Downloaded files will get
	//     the current local system date/time and no attempt will be made to try to
	//     preserver the last-modified date/time of the file on the server.
	// 
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "OpenNonExclusive" - Introduced in v9.5.0.78. When downloading files on
	//     Windows systems, open the local file with non-exclusive access to allow other
	//     programs the ability to access the file as it's being downloaded.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "EnableTls13" - Introduced in v9.5.0.82. Causes TLS 1.3 to be offered in the
	//     ClientHello of the TLS protocol, allowing the server to select TLS 1.3 for the
	//     session. Future versions of Chilkat will enable TLS 1.3 by default. This option
	//     is only necessary in v9.5.0.82 if TLS 1.3 is desired.
	//     "DisableTls13" - Disables the use of TLS 1.3. TLS 1.3 is enabled by default
	//     in Chilkat v9.5.0.84 and above. This keyword can be used to avoid TLS 1.3 if it
	//     causes problems.
	//     "NoPreserveFileTime" - Introduced in v9.5.0.85. Downloaded files will get
	//     the current local system date/time and no attempt will be made to try to
	//     preserver the last-modified date/time of the file on the server.
	// 
	void put_UncommonOptions(const char *newVal);

	// The average upload rate in bytes/second. This property is updated in real-time
	// during any FTP upload (asynchronous or synchronous).
	int get_UploadTransferRate(void);

	// If true, the FTP2 component will use the EPSV command instead of PASV for
	// passive mode data transfers. The default value of this property is false. (It
	// is somewhat uncommon for FTP servers to support EPSV.)
	// 
	// Note: If the AutoFeat property is true, then the FTP server's features are
	// automatically queried after connecting. In this case, if the AutoSetUseEpsv
	// property is also set to true, the UseEpsv property is automatically set to
	// true if the FTP server supports EPSV.
	// 
	// Important: EPSV can cause problems with some deep-inspection firewalls. If a
	// passive data connection cannot be established, make sure to test with both the
	// AutoSetUseEpsv and UseEpsv properties set equal to false.
	// 
	bool get_UseEpsv(void);
	// If true, the FTP2 component will use the EPSV command instead of PASV for
	// passive mode data transfers. The default value of this property is false. (It
	// is somewhat uncommon for FTP servers to support EPSV.)
	// 
	// Note: If the AutoFeat property is true, then the FTP server's features are
	// automatically queried after connecting. In this case, if the AutoSetUseEpsv
	// property is also set to true, the UseEpsv property is automatically set to
	// true if the FTP server supports EPSV.
	// 
	// Important: EPSV can cause problems with some deep-inspection firewalls. If a
	// passive data connection cannot be established, make sure to test with both the
	// AutoSetUseEpsv and UseEpsv properties set equal to false.
	// 
	void put_UseEpsv(bool newVal);

	// Username for logging into the FTP server. Defaults to "anonymous".
	void get_Username(CkString &str);
	// Username for logging into the FTP server. Defaults to "anonymous".
	const char *username(void);
	// Username for logging into the FTP server. Defaults to "anonymous".
	void put_Username(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Same as PutFile but the file on the FTP server is appended.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool AppendFile(const char *localFilePath, const char *remoteFilePath);

	// Same as PutFile but the file on the FTP server is appended.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *AppendFileAsync(const char *localFilePath, const char *remoteFilePath);


	// Same as PutFileFromBinaryData, except the file on the FTP server is appended.
	bool AppendFileFromBinaryData(const char *remoteFilename, CkByteData &content);

	// Same as PutFileFromBinaryData, except the file on the FTP server is appended.
	CkTask *AppendFileFromBinaryDataAsync(const char *remoteFilename, CkByteData &content);


	// Same as PutFileFromTextData, except the file on the FTP server is appended.
	bool AppendFileFromTextData(const char *remoteFilename, const char *textData, const char *charset);

	// Same as PutFileFromTextData, except the file on the FTP server is appended.
	CkTask *AppendFileFromTextDataAsync(const char *remoteFilename, const char *textData, const char *charset);


	// Changes the current remote directory. The remoteDirPath should be relative to the current
	// remote directory, which is initially the HOME directory of the FTP user account.
	// 
	// If the remoteDirPath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool ChangeRemoteDir(const char *remoteDirPath);

	// Changes the current remote directory. The remoteDirPath should be relative to the current
	// remote directory, which is initially the HOME directory of the FTP user account.
	// 
	// If the remoteDirPath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *ChangeRemoteDirAsync(const char *remoteDirPath);


	// Returns true if currently connected and logged into an FTP server, otherwise
	// returns false.
	// 
	// Note: This may cause a NOOP command to be sent to the FTP server.
	// 
	bool CheckConnection(void);

	// Returns true if currently connected and logged into an FTP server, otherwise
	// returns false.
	// 
	// Note: This may cause a NOOP command to be sent to the FTP server.
	// 
	CkTask *CheckConnectionAsync(void);


	// Reverts the FTP control channel from SSL/TLS to an unencrypted channel. This may
	// be required when using FTPS with AUTH TLS where the FTP client is behind a DSL
	// or cable-modem router that performs NAT (network address translation). If the
	// control channel is encrypted, the router is unable to translate the IP address
	// sent in the PORT command for data transfers. By clearing the control channel,
	// the data transfers will remain encrypted, but the FTP commands are passed
	// unencrypted. Your program would typically clear the control channel after
	// authenticating.
	bool ClearControlChannel(void);

	// Reverts the FTP control channel from SSL/TLS to an unencrypted channel. This may
	// be required when using FTPS with AUTH TLS where the FTP client is behind a DSL
	// or cable-modem router that performs NAT (network address translation). If the
	// control channel is encrypted, the router is unable to translate the IP address
	// sent in the PORT command for data transfers. By clearing the control channel,
	// the data transfers will remain encrypted, but the FTP commands are passed
	// unencrypted. Your program would typically clear the control channel after
	// authenticating.
	CkTask *ClearControlChannelAsync(void);


	// The GetDirCount method returns the count of files and sub-directories in the
	// current remote FTP directory, according to the ListPattern property. For
	// example, if ListPattern is set to "*.xml", then GetDirCount returns the count of
	// XML files in the remote directory.
	// 
	// The 1st time it is accessed, the component will (behind the scenes) fetch the
	// directory listing from the FTP server. This information is cached in the
	// component until (1) the current remote directory is changed, or (2) the
	// ListPattern is changed, or (3) the this method (ClearDirCache) is called.
	// 
	void ClearDirCache(void);


	// Clears the in-memory session log.
	void ClearSessionLog(void);


	// Connects and logs in to the FTP server using the username/password provided in
	// the component properties. Check the integer value of the ConnectFailReason if
	// this method returns false (indicating failure).
	// 
	// Note: To separately establish the connection and then authenticate (in separate
	// method calls), call ConnectOnly followed by LoginAfterConnectOnly.
	// 
	// Important: All TCP-based Internet communications, regardless of the protocol
	// (such as HTTP, FTP, SSH, IMAP, POP3, SMTP, etc.), and regardless of SSL/TLS,
	// begin with establishing a TCP connection to a remote host:port. External
	// security-related infrastructure such as software firewalls (Windows Firewall),
	// hardware firewalls, anti-virus, at either source or destination (or both) can
	// block the connection. If the connection fails, make sure to check all potential
	// external causes of blockage.
	// 
	bool Connect(void);

	// Connects and logs in to the FTP server using the username/password provided in
	// the component properties. Check the integer value of the ConnectFailReason if
	// this method returns false (indicating failure).
	// 
	// Note: To separately establish the connection and then authenticate (in separate
	// method calls), call ConnectOnly followed by LoginAfterConnectOnly.
	// 
	// Important: All TCP-based Internet communications, regardless of the protocol
	// (such as HTTP, FTP, SSH, IMAP, POP3, SMTP, etc.), and regardless of SSL/TLS,
	// begin with establishing a TCP connection to a remote host:port. External
	// security-related infrastructure such as software firewalls (Windows Firewall),
	// hardware firewalls, anti-virus, at either source or destination (or both) can
	// block the connection. If the connection fails, make sure to check all potential
	// external causes of blockage.
	// 
	CkTask *ConnectAsync(void);


	// Connects to the FTP server, but does not authenticate. The combination of
	// calling this method followed by LoginAfterConnectOnly is the equivalent of
	// calling the Connect method (which both connects and authenticates).
	// 
	// Important: All TCP-based Internet communications, regardless of the protocol
	// (such as HTTP, FTP, SSH, IMAP, POP3, SMTP, etc.), and regardless of SSL/TLS,
	// begin with establishing a TCP connection to a remote host:port. External
	// security-related infrastructure such as software firewalls (Windows Firewall),
	// hardware firewalls, anti-virus, at either source or destination (or both) can
	// block the connection. If the connection fails, make sure to check all potential
	// external causes of blockage.
	// 
	bool ConnectOnly(void);

	// Connects to the FTP server, but does not authenticate. The combination of
	// calling this method followed by LoginAfterConnectOnly is the equivalent of
	// calling the Connect method (which both connects and authenticates).
	// 
	// Important: All TCP-based Internet communications, regardless of the protocol
	// (such as HTTP, FTP, SSH, IMAP, POP3, SMTP, etc.), and regardless of SSL/TLS,
	// begin with establishing a TCP connection to a remote host:port. External
	// security-related infrastructure such as software firewalls (Windows Firewall),
	// hardware firewalls, anti-virus, at either source or destination (or both) can
	// block the connection. If the connection fails, make sure to check all potential
	// external causes of blockage.
	// 
	CkTask *ConnectOnlyAsync(void);


	// Explicitly converts the control channel to a secure SSL/TLS connection.
	// 
	// Note: If you initially connect with either the AuthTls or AuthSsl property set
	// to true, then DO NOT call ConvertToTls. The control channel is automatically
	// converted to SSL/TLS from within the Connect method when these properties are
	// set.
	// 
	// Note: It is very uncommon for this method to be needed.
	// 
	bool ConvertToTls(void);

	// Explicitly converts the control channel to a secure SSL/TLS connection.
	// 
	// Note: If you initially connect with either the AuthTls or AuthSsl property set
	// to true, then DO NOT call ConvertToTls. The control channel is automatically
	// converted to SSL/TLS from within the Connect method when these properties are
	// set.
	// 
	// Note: It is very uncommon for this method to be needed.
	// 
	CkTask *ConvertToTlsAsync(void);


	// Creates an "FTP plan" that lists the FTP operations that would be performed when
	// PutTree is called. Additionally, the PutPlan method executes an "FTP plan" and
	// logs each successful operation to a plan log file. If a large-scale upload is
	// interrupted, the PutPlan can be resumed, skipping over the operations already
	// listed in the plan log file.
	bool CreatePlan(const char *localDir, CkString &outStr);

	// Creates an "FTP plan" that lists the FTP operations that would be performed when
	// PutTree is called. Additionally, the PutPlan method executes an "FTP plan" and
	// logs each successful operation to a plan log file. If a large-scale upload is
	// interrupted, the PutPlan can be resumed, skipping over the operations already
	// listed in the plan log file.
	const char *createPlan(const char *localDir);
	// Creates an "FTP plan" that lists the FTP operations that would be performed when
	// PutTree is called. Additionally, the PutPlan method executes an "FTP plan" and
	// logs each successful operation to a plan log file. If a large-scale upload is
	// interrupted, the PutPlan can be resumed, skipping over the operations already
	// listed in the plan log file.
	CkTask *CreatePlanAsync(const char *localDir);


	// Creates a directory on the FTP server. If the directory already exists, a new
	// one is not created and false is returned.
	// 
	// If the remoteDirPath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool CreateRemoteDir(const char *remoteDirPath);

	// Creates a directory on the FTP server. If the directory already exists, a new
	// one is not created and false is returned.
	// 
	// If the remoteDirPath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *CreateRemoteDirAsync(const char *remoteDirPath);


	// Deletes all the files in the current remote FTP directory matching the pattern.
	// Returns the number of files deleted, or -1 for failure. The pattern is a string
	// such as "*.txt", where any number of "*" wildcard characters can be used. "*"
	// matches 0 or more of any character.
	int DeleteMatching(const char *remotePattern);

	// Deletes all the files in the current remote FTP directory matching the pattern.
	// Returns the number of files deleted, or -1 for failure. The pattern is a string
	// such as "*.txt", where any number of "*" wildcard characters can be used. "*"
	// matches 0 or more of any character.
	CkTask *DeleteMatchingAsync(const char *remotePattern);


	// Deletes a file on the FTP server.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool DeleteRemoteFile(const char *remoteFilePath);

	// Deletes a file on the FTP server.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *DeleteRemoteFileAsync(const char *remoteFilePath);


	// Deletes the entire subtree and all files from the current remote FTP directory.
	// To delete a subtree on the FTP server, your program would first navigate to the
	// root of the subtree to be deleted by calling ChangeRemoteDir, and then call
	// DeleteTree. There are two event callbacks: VerifyDeleteFile and VerifyDeleteDir.
	// Both are called prior to deleting each file or directory. The arguments to the
	// callback include the full filepath of the file or directory, and an output-only
	// "skip" flag. If your application sets the skip flag to true, the file or
	// directory is NOT deleted. If a directory is not deleted, all files and
	// sub-directories will remain. Example programs can be found at
	// http://www.example-code.com/
	bool DeleteTree(void);

	// Deletes the entire subtree and all files from the current remote FTP directory.
	// To delete a subtree on the FTP server, your program would first navigate to the
	// root of the subtree to be deleted by calling ChangeRemoteDir, and then call
	// DeleteTree. There are two event callbacks: VerifyDeleteFile and VerifyDeleteDir.
	// Both are called prior to deleting each file or directory. The arguments to the
	// callback include the full filepath of the file or directory, and an output-only
	// "skip" flag. If your application sets the skip flag to true, the file or
	// directory is NOT deleted. If a directory is not deleted, all files and
	// sub-directories will remain. Example programs can be found at
	// http://www.example-code.com/
	CkTask *DeleteTreeAsync(void);


	// Automatically determines the ProxyMethod that should be used with an FTP proxy
	// server. Tries each of the five possible ProxyMethod settings and returns the
	// value (1-5) of the ProxyMethod that succeeded.
	// 
	// This method may take a minute or two to complete. Returns 0 if no proxy methods
	// were successful. Returns -1 to indicate an error (i.e. it was unable to test all
	// proxy methods.)
	// 
	int DetermineProxyMethod(void);

	// Automatically determines the ProxyMethod that should be used with an FTP proxy
	// server. Tries each of the five possible ProxyMethod settings and returns the
	// value (1-5) of the ProxyMethod that succeeded.
	// 
	// This method may take a minute or two to complete. Returns 0 if no proxy methods
	// were successful. Returns -1 to indicate an error (i.e. it was unable to test all
	// proxy methods.)
	// 
	CkTask *DetermineProxyMethodAsync(void);


	// Discovers which combinations of FTP2 property settings result in successful data
	// transfers.
	// 
	// DetermineSettings tries 13 different combinations of these properties:
	// Ssl
	// AuthTls
	// AuthSsl
	// Port
	// Passive
	// PassiveUseHostAddr
	// Within the FTP protocol, the process of fetching a directory listing is also
	// considered a "data transfer". The DetermineSettings method works by checking to
	// see which combinations result in a successful directory listing download. The
	// method takes no arguments and returns a string containing an XML report of the
	// results. It is a blocking call that may take approximately a minute to run. If
	// you are unsure about how to interpret the results, cut-and-paste it into an
	// email and send it to support@chilkatsoft.com.
	// 
	bool DetermineSettings(CkString &outXmlReport);

	// Discovers which combinations of FTP2 property settings result in successful data
	// transfers.
	// 
	// DetermineSettings tries 13 different combinations of these properties:
	// Ssl
	// AuthTls
	// AuthSsl
	// Port
	// Passive
	// PassiveUseHostAddr
	// Within the FTP protocol, the process of fetching a directory listing is also
	// considered a "data transfer". The DetermineSettings method works by checking to
	// see which combinations result in a successful directory listing download. The
	// method takes no arguments and returns a string containing an XML report of the
	// results. It is a blocking call that may take approximately a minute to run. If
	// you are unsure about how to interpret the results, cut-and-paste it into an
	// email and send it to support@chilkatsoft.com.
	// 
	const char *determineSettings(void);
	// Discovers which combinations of FTP2 property settings result in successful data
	// transfers.
	// 
	// DetermineSettings tries 13 different combinations of these properties:
	// Ssl
	// AuthTls
	// AuthSsl
	// Port
	// Passive
	// PassiveUseHostAddr
	// Within the FTP protocol, the process of fetching a directory listing is also
	// considered a "data transfer". The DetermineSettings method works by checking to
	// see which combinations result in a successful directory listing download. The
	// method takes no arguments and returns a string containing an XML report of the
	// results. It is a blocking call that may take approximately a minute to run. If
	// you are unsure about how to interpret the results, cut-and-paste it into an
	// email and send it to support@chilkatsoft.com.
	// 
	CkTask *DetermineSettingsAsync(void);


	// Recursively downloads the structure of a complete remote directory tree. Returns
	// an XML document with the directory structure.
	// 
	// Note: Starting in version 9.5.0.47, the SyncMustMatch and SyncMustNotMatch
	// properties apply to this method.
	// 
	bool DirTreeXml(CkString &outStrXml);

	// Recursively downloads the structure of a complete remote directory tree. Returns
	// an XML document with the directory structure.
	// 
	// Note: Starting in version 9.5.0.47, the SyncMustMatch and SyncMustNotMatch
	// properties apply to this method.
	// 
	const char *dirTreeXml(void);
	// Recursively downloads the structure of a complete remote directory tree. Returns
	// an XML document with the directory structure.
	// 
	// Note: Starting in version 9.5.0.47, the SyncMustMatch and SyncMustNotMatch
	// properties apply to this method.
	// 
	CkTask *DirTreeXmlAsync(void);


	// Disconnects from the FTP server, ending the current session.
	bool Disconnect(void);

	// Disconnects from the FTP server, ending the current session.
	CkTask *DisconnectAsync(void);


	// Downloads an entire tree from the FTP server and recreates the directory tree on
	// the local filesystem.
	// 
	// This method downloads all the files and subdirectories in the current remote
	// directory. An application would first navigate to the directory to be downloaded
	// via ChangeRemoteDir and then call this method.
	// 
	// Note: Starting in version 9.5.0.47, the SyncMustMatch and SyncMustNotMatch
	// properties apply to this method.
	// 
	bool DownloadTree(const char *localRoot);

	// Downloads an entire tree from the FTP server and recreates the directory tree on
	// the local filesystem.
	// 
	// This method downloads all the files and subdirectories in the current remote
	// directory. An application would first navigate to the directory to be downloaded
	// via ChangeRemoteDir and then call this method.
	// 
	// Note: Starting in version 9.5.0.47, the SyncMustMatch and SyncMustNotMatch
	// properties apply to this method.
	// 
	CkTask *DownloadTreeAsync(const char *localRoot);


	// Sends a FEAT command to the FTP server and returns the response. Returns a
	// zero-length string to indicate failure. Here is a typical response:
	// 211-Features:
	//  MDTM
	//  REST STREAM
	//  SIZE
	//  MLST type*;size*;modify*;
	//  MLSD
	//  AUTH SSL
	//  AUTH TLS
	//  UTF8
	//  CLNT
	//  MFMT
	// 211 End
	bool Feat(CkString &outStr);

	// Sends a FEAT command to the FTP server and returns the response. Returns a
	// zero-length string to indicate failure. Here is a typical response:
	// 211-Features:
	//  MDTM
	//  REST STREAM
	//  SIZE
	//  MLST type*;size*;modify*;
	//  MLSD
	//  AUTH SSL
	//  AUTH TLS
	//  UTF8
	//  CLNT
	//  MFMT
	// 211 End
	const char *feat(void);
	// Sends a FEAT command to the FTP server and returns the response. Returns a
	// zero-length string to indicate failure. Here is a typical response:
	// 211-Features:
	//  MDTM
	//  REST STREAM
	//  SIZE
	//  MLST type*;size*;modify*;
	//  MLSD
	//  AUTH SSL
	//  AUTH TLS
	//  UTF8
	//  CLNT
	//  MFMT
	// 211 End
	CkTask *FeatAsync(void);


	// Returns the create date/time for the Nth file or sub-directory in the current
	// remote directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	// The caller is responsible for deleting the object returned by this method.
	CkDateTime *GetCreateDt(int index);

	// Returns the create date/time for the Nth file or sub-directory in the current
	// remote directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	CkTask *GetCreateDtAsync(int index);


	// Returns the file-creation date/time for a remote file by filename.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	// Note: Linux/Unix type filesystems do not store "create" date/times. Therefore,
	// if the FTP server is on such as system, this method will return a date/time
	// equal to the last-modified date/time.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkDateTime *GetCreateDtByName(const char *filename);

	// Returns the file-creation date/time for a remote file by filename.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	// Note: Linux/Unix type filesystems do not store "create" date/times. Therefore,
	// if the FTP server is on such as system, this method will return a date/time
	// equal to the last-modified date/time.
	// 
	CkTask *GetCreateDtByNameAsync(const char *filename);


	// Returns the create time for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	// 
	// Note: The FILETIME is a Windows-based format. See
	// http://support.microsoft.com/kb/188768 for more information.
	// 
	bool GetCreateFTime(int index, FILETIME &outFileTime);


	// Returns the create time for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	bool GetCreateTime(int index, SYSTEMTIME &outSysTime);


	// Returns the file-creation date/time for a remote file by filename.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	// Note: Linux/Unix type filesystems do not store "create" date/times. If the FTP
	// server is on such as system, this method will return a date/time equal to the
	// last-modified date/time.
	// 
	bool GetCreateTimeByName(const char *filename, SYSTEMTIME &outSysTime);


	// Returns the file-creation date/time (in RFC822 string format, such as "Tue, 25
	// Sep 2012 12:25:32 -0500") for a remote file by filename.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	// Note: Linux/Unix type filesystems do not store "create" date/times. If the FTP
	// server is on such as system, this method will return a date/time equal to the
	// last-modified date/time.
	// 
	bool GetCreateTimeByNameStr(const char *filename, CkString &outStr);

	// Returns the file-creation date/time (in RFC822 string format, such as "Tue, 25
	// Sep 2012 12:25:32 -0500") for a remote file by filename.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	// Note: Linux/Unix type filesystems do not store "create" date/times. If the FTP
	// server is on such as system, this method will return a date/time equal to the
	// last-modified date/time.
	// 
	const char *getCreateTimeByNameStr(const char *filename);
	// Returns the file-creation date/time (in RFC822 string format, such as "Tue, 25
	// Sep 2012 12:25:32 -0500") for a remote file by filename.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	// Note: Linux/Unix type filesystems do not store "create" date/times. If the FTP
	// server is on such as system, this method will return a date/time equal to the
	// last-modified date/time.
	// 
	const char *createTimeByNameStr(const char *filename);

	// Returns the file-creation date/time (in RFC822 string format, such as "Tue, 25
	// Sep 2012 12:25:32 -0500") for a remote file by filename.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	// Note: Linux/Unix type filesystems do not store "create" date/times. If the FTP
	// server is on such as system, this method will return a date/time equal to the
	// last-modified date/time.
	// 
	CkTask *GetCreateTimeByNameStrAsync(const char *filename);


	// Returns the create time (in RFC822 string format, such as "Tue, 25 Sep 2012
	// 12:25:32 -0500") for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	bool GetCreateTimeStr(int index, CkString &outStr);

	// Returns the create time (in RFC822 string format, such as "Tue, 25 Sep 2012
	// 12:25:32 -0500") for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	const char *getCreateTimeStr(int index);
	// Returns the create time (in RFC822 string format, such as "Tue, 25 Sep 2012
	// 12:25:32 -0500") for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	const char *createTimeStr(int index);

	// Returns the create time (in RFC822 string format, such as "Tue, 25 Sep 2012
	// 12:25:32 -0500") for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	CkTask *GetCreateTimeStrAsync(int index);


	// Returns the current remote directory.
	bool GetCurrentRemoteDir(CkString &outStr);

	// Returns the current remote directory.
	const char *getCurrentRemoteDir(void);
	// Returns the current remote directory.
	const char *currentRemoteDir(void);

	// Returns the current remote directory.
	CkTask *GetCurrentRemoteDirAsync(void);


	// Returns the number of files and sub-directories in the current remote directory
	// that match the ListPattern property.
	// 
	// Important: Calling this method may cause the directory listing to be retrieved
	// from the FTP server. For FTP servers that do not support the MLST/MLSD commands,
	// this is technically a data transfer that requires a temporary data connection to
	// be established in the same way as when uploading or downloading files. If your
	// program hangs while calling this method, it probably means that the data
	// connection could not be established. The most common solution is to switch to
	// using Passive mode by setting the Passive property = true, with the
	// PassiveUseHostAddr property also set equal to true. If this does not help,
	// examine the contents of the LastErrorText property after this method finally
	// returns (after timing out). Also, seethis Chilkat blog post about FTP connection
	// settings
	// <http://www.cknotes.com/?p=282>.
	// 
	int GetDirCount(void);

	// Returns the number of files and sub-directories in the current remote directory
	// that match the ListPattern property.
	// 
	// Important: Calling this method may cause the directory listing to be retrieved
	// from the FTP server. For FTP servers that do not support the MLST/MLSD commands,
	// this is technically a data transfer that requires a temporary data connection to
	// be established in the same way as when uploading or downloading files. If your
	// program hangs while calling this method, it probably means that the data
	// connection could not be established. The most common solution is to switch to
	// using Passive mode by setting the Passive property = true, with the
	// PassiveUseHostAddr property also set equal to true. If this does not help,
	// examine the contents of the LastErrorText property after this method finally
	// returns (after timing out). Also, seethis Chilkat blog post about FTP connection
	// settings
	// <http://www.cknotes.com/?p=282>.
	// 
	CkTask *GetDirCountAsync(void);


	// Downloads a file from the FTP server to the local filesystem.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool GetFile(const char *remoteFilePath, const char *localFilePath);

	// Downloads a file from the FTP server to the local filesystem.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *GetFileAsync(const char *remoteFilePath, const char *localFilePath);


	// Downloads a file from the FTP server into a BinData object.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool GetFileBd(const char *remoteFilePath, CkBinData &binData);

	// Downloads a file from the FTP server into a BinData object.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *GetFileBdAsync(const char *remoteFilePath, CkBinData &binData);


	// Returns the filename for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	bool GetFilename(int index, CkString &outStr);

	// Returns the filename for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	const char *getFilename(int index);
	// Returns the filename for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	const char *filename(int index);

	// Returns the filename for the Nth file or sub-directory in the current remote
	// directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	CkTask *GetFilenameAsync(int index);


	// Downloads a file from the FTP server into a StringBuilder object.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool GetFileSb(const char *remoteFilePath, const char *charset, CkStringBuilder &sb);

	// Downloads a file from the FTP server into a StringBuilder object.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *GetFileSbAsync(const char *remoteFilePath, const char *charset, CkStringBuilder &sb);


	// Downloads a file to a stream. If called synchronously, the remoteFilePath must have a
	// sink, such as a file or another stream object. If called asynchronously, then
	// the foreground thread can read the stream.
	bool GetFileToStream(const char *remoteFilePath, CkStream &toStream);

	// Downloads a file to a stream. If called synchronously, the remoteFilePath must have a
	// sink, such as a file or another stream object. If called asynchronously, then
	// the foreground thread can read the stream.
	CkTask *GetFileToStreamAsync(const char *remoteFilePath, CkStream &toStream);


	// Returns group name, if available, for the Nth file. If empty, then no group
	// information is available.
	// 
	// Note: When MLSD is used to get directory listings, it is likely that the owner
	// and group information is not transmitted. In cases where the FTP server is on a
	// UNIX/Linux system, the AllowMlsd property can be set to false to force UNIX
	// directory listings instead of MLSD directory listings. This should result in
	// being able to obtain owner/group information. However, it may sacrifice the
	// quality and accuracy of the various date/time values that are returned.
	// 
	bool GetGroup(int index, CkString &outStr);

	// Returns group name, if available, for the Nth file. If empty, then no group
	// information is available.
	// 
	// Note: When MLSD is used to get directory listings, it is likely that the owner
	// and group information is not transmitted. In cases where the FTP server is on a
	// UNIX/Linux system, the AllowMlsd property can be set to false to force UNIX
	// directory listings instead of MLSD directory listings. This should result in
	// being able to obtain owner/group information. However, it may sacrifice the
	// quality and accuracy of the various date/time values that are returned.
	// 
	const char *getGroup(int index);
	// Returns group name, if available, for the Nth file. If empty, then no group
	// information is available.
	// 
	// Note: When MLSD is used to get directory listings, it is likely that the owner
	// and group information is not transmitted. In cases where the FTP server is on a
	// UNIX/Linux system, the AllowMlsd property can be set to false to force UNIX
	// directory listings instead of MLSD directory listings. This should result in
	// being able to obtain owner/group information. However, it may sacrifice the
	// quality and accuracy of the various date/time values that are returned.
	// 
	const char *group(int index);

	// Returns group name, if available, for the Nth file. If empty, then no group
	// information is available.
	// 
	// Note: When MLSD is used to get directory listings, it is likely that the owner
	// and group information is not transmitted. In cases where the FTP server is on a
	// UNIX/Linux system, the AllowMlsd property can be set to false to force UNIX
	// directory listings instead of MLSD directory listings. This should result in
	// being able to obtain owner/group information. However, it may sacrifice the
	// quality and accuracy of the various date/time values that are returned.
	// 
	CkTask *GetGroupAsync(int index);


	// Returns true for a sub-directory and false for a file, for the Nth entry in
	// the current remote directory. The first file/dir is at index 0, and the last one
	// is at index (GetDirCount()-1)
	bool GetIsDirectory(int index);

	// Returns true for a sub-directory and false for a file, for the Nth entry in
	// the current remote directory. The first file/dir is at index 0, and the last one
	// is at index (GetDirCount()-1)
	CkTask *GetIsDirectoryAsync(int index);


	// Returns true if the remote file is a symbolic link. (Symbolic links only exist
	// on Unix/Linux systems, not on Windows filesystems.)
	bool GetIsSymbolicLink(int index);

	// Returns true if the remote file is a symbolic link. (Symbolic links only exist
	// on Unix/Linux systems, not on Windows filesystems.)
	CkTask *GetIsSymbolicLinkAsync(int index);


	// Returns the last modified date/time for the Nth file or sub-directory in the
	// current remote directory. The first file/dir is at index 0, and the last one is
	// at index (GetDirCount()-1)
	// The caller is responsible for deleting the object returned by this method.
	CkDateTime *GetLastModDt(int index);

	// Returns the last modified date/time for the Nth file or sub-directory in the
	// current remote directory. The first file/dir is at index 0, and the last one is
	// at index (GetDirCount()-1)
	CkTask *GetLastModDtAsync(int index);


	// Returns the last-modified date/time for a remote file.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkDateTime *GetLastModDtByName(const char *filename);

	// Returns the last-modified date/time for a remote file.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	CkTask *GetLastModDtByNameAsync(const char *filename);


	// Returns the last modified date/time for the Nth file or sub-directory in the
	// current remote directory. The first file/dir is at index 0, and the last one is
	// at index (GetDirCount()-1)
	// 
	// Note: The FILETIME is a Windows-based format. See
	// http://support.microsoft.com/kb/188768 for more information.
	// 
	bool GetLastModifiedFTime(int index, FILETIME &outFileTime);


	// Returns the last modified date/time for the Nth file or sub-directory in the
	// current remote directory. The first file/dir is at index 0, and the last one is
	// at index (GetDirCount()-1)
	bool GetLastModifiedTime(int index, SYSTEMTIME &outSysTime);


	// Returns the last-modified date/time for a remote file.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	bool GetLastModifiedTimeByName(const char *filename, SYSTEMTIME &outSysTime);


	// Returns a remote file's last-modified date/time in RFC822 string format, such as
	// "Tue, 25 Sep 2012 12:25:32 -0500".
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	bool GetLastModifiedTimeByNameStr(const char *filename, CkString &outStr);

	// Returns a remote file's last-modified date/time in RFC822 string format, such as
	// "Tue, 25 Sep 2012 12:25:32 -0500".
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	const char *getLastModifiedTimeByNameStr(const char *filename);
	// Returns a remote file's last-modified date/time in RFC822 string format, such as
	// "Tue, 25 Sep 2012 12:25:32 -0500".
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	const char *lastModifiedTimeByNameStr(const char *filename);

	// Returns a remote file's last-modified date/time in RFC822 string format, such as
	// "Tue, 25 Sep 2012 12:25:32 -0500".
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	CkTask *GetLastModifiedTimeByNameStrAsync(const char *filename);


	// Returns the last modified date/time (in RFC822 string format, such as "Tue, 25
	// Sep 2012 12:25:32 -0500") for the Nth file or sub-directory in the current
	// remote directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	bool GetLastModifiedTimeStr(int index, CkString &outStr);

	// Returns the last modified date/time (in RFC822 string format, such as "Tue, 25
	// Sep 2012 12:25:32 -0500") for the Nth file or sub-directory in the current
	// remote directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	const char *getLastModifiedTimeStr(int index);
	// Returns the last modified date/time (in RFC822 string format, such as "Tue, 25
	// Sep 2012 12:25:32 -0500") for the Nth file or sub-directory in the current
	// remote directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	const char *lastModifiedTimeStr(int index);

	// Returns the last modified date/time (in RFC822 string format, such as "Tue, 25
	// Sep 2012 12:25:32 -0500") for the Nth file or sub-directory in the current
	// remote directory. The first file/dir is at index 0, and the last one is at index
	// (GetDirCount()-1)
	CkTask *GetLastModifiedTimeStrAsync(int index);


	// Returns owner name, if available, for the Nth file. If empty, then no owner
	// information is available.
	// 
	// Note: When MLSD is used to get directory listings, it is likely that the owner
	// and group information is not transmitted. In cases where the FTP server is on a
	// UNIX/Linux system, the AllowMlsd property can be set to false to force UNIX
	// directory listings instead of MLSD directory listings. This should result in
	// being able to obtain owner/group information. However, it may sacrifice the
	// quality and accuracy of the various date/time values that are returned.
	// 
	bool GetOwner(int index, CkString &outStr);

	// Returns owner name, if available, for the Nth file. If empty, then no owner
	// information is available.
	// 
	// Note: When MLSD is used to get directory listings, it is likely that the owner
	// and group information is not transmitted. In cases where the FTP server is on a
	// UNIX/Linux system, the AllowMlsd property can be set to false to force UNIX
	// directory listings instead of MLSD directory listings. This should result in
	// being able to obtain owner/group information. However, it may sacrifice the
	// quality and accuracy of the various date/time values that are returned.
	// 
	const char *getOwner(int index);
	// Returns owner name, if available, for the Nth file. If empty, then no owner
	// information is available.
	// 
	// Note: When MLSD is used to get directory listings, it is likely that the owner
	// and group information is not transmitted. In cases where the FTP server is on a
	// UNIX/Linux system, the AllowMlsd property can be set to false to force UNIX
	// directory listings instead of MLSD directory listings. This should result in
	// being able to obtain owner/group information. However, it may sacrifice the
	// quality and accuracy of the various date/time values that are returned.
	// 
	const char *owner(int index);

	// Returns owner name, if available, for the Nth file. If empty, then no owner
	// information is available.
	// 
	// Note: When MLSD is used to get directory listings, it is likely that the owner
	// and group information is not transmitted. In cases where the FTP server is on a
	// UNIX/Linux system, the AllowMlsd property can be set to false to force UNIX
	// directory listings instead of MLSD directory listings. This should result in
	// being able to obtain owner/group information. However, it may sacrifice the
	// quality and accuracy of the various date/time values that are returned.
	// 
	CkTask *GetOwnerAsync(int index);


	// Returns permissions information, if available, for the Nth file. If empty, then
	// no permissions information is available. The value returned by the GetPermType
	// method defines the content and format of the permissions string returned by this
	// method. Possible permission types are "mlsd", "unix", "netware", "openvms", and
	// "batchStatusFlags". The format of each permission type is as follows:
	// --------------------------------------------------------------------------------
	// 
	// PermType: mlsd:
	// 
	// A "perm fact" is returned. The format of the perm fact is defined in RFC 3659 as
	// follows:
	//   The perm fact is used to indicate access rights the current FTP user
	//    has over the object listed.  Its value is always an unordered
	//    sequence of alphabetic characters.
	// 
	//       perm-fact    = "Perm" "=" *pvals
	//       pvals        = "a" / "c" / "d" / "e" / "f" /
	//                      "l" / "m" / "p" / "r" / "w"
	// 
	//    There are ten permission indicators currently defined.  Many are
	//    meaningful only when used with a particular type of object.  The
	//    indicators are case independent, "d" and "D" are the same indicator.
	// 
	//    The "a" permission applies to objects of type=file, and indicates
	//    that the APPE (append) command may be applied to the file named.
	// 
	//    The "c" permission applies to objects of type=dir (and type=pdir,
	//    type=cdir).  It indicates that files may be created in the directory
	//    named.  That is, that a STOU command is likely to succeed, and that
	//    STOR and APPE commands might succeed if the file named did not
	//    previously exist, but is to be created in the directory object that
	//    has the "c" permission.  It also indicates that the RNTO command is
	//    likely to succeed for names in the directory.
	// 
	//    The "d" permission applies to all types.  It indicates that the
	//    object named may be deleted, that is, that the RMD command may be
	//    applied to it if it is a directory, and otherwise that the DELE
	//    command may be applied to it.
	// 
	//    The "e" permission applies to the directory types.  When set on an
	//    object of type=dir, type=cdir, or type=pdir it indicates that a CWD
	//    command naming the object should succeed, and the user should be able
	//    to enter the directory named.  For type=pdir it also indicates that
	//    the CDUP command may succeed (if this particular pathname is the one
	//    to which a CDUP would apply.)
	// 
	//    The "f" permission for objects indicates that the object named may be
	//    renamed - that is, may be the object of an RNFR command.
	// 
	//    The "l" permission applies to the directory file types, and indicates
	//    that the listing commands, LIST, NLST, and MLSD may be applied to the
	//    directory in question.
	// 
	//    The "m" permission applies to directory types, and indicates that the
	//    MKD command may be used to create a new directory within the
	//    directory under consideration.
	// 
	//    The "p" permission applies to directory types, and indicates that
	//    objects in the directory may be deleted, or (stretching naming a
	//    little) that the directory may be purged.  Note: it does not indicate
	//    that the RMD command may be used to remove the directory named
	//    itself, the "d" permission indicator indicates that.
	// 
	//    The "r" permission applies to type=file objects, and for some
	//    systems, perhaps to other types of objects, and indicates that the
	//    RETR command may be applied to that object.
	// 
	//    The "w" permission applies to type=file objects, and for some
	//    systems, perhaps to other types of objects, and indicates that the
	//    STOR command may be applied to the object named.
	// 
	//    Note: That a permission indicator is set can never imply that the
	//       appropriate command is guaranteed to work -- just that it might.
	//       Other system specific limitations, such as limitations on
	//       available space for storing files, may cause an operation to fail,
	//       where the permission flags may have indicated that it was likely
	//       to succeed.  The permissions are a guide only.
	// 
	//    Implementation note: The permissions are described here as they apply
	//       to FTP commands.  They may not map easily into particular
	//       permissions available on the server's operating system.  Servers
	//       are expected to synthesize these permission bits from the
	//       permission information available from operating system.  For
	//       example, to correctly determine whether the "D" permission bit
	//       should be set on a directory for a server running on the UNIX(TM)
	//       operating system, the server should check that the directory named
	//       is empty, and that the user has write permission on both the
	//       directory under consideration, and its parent directory.
	// 
	//       Some systems may have more specific permissions than those listed
	//       here, such systems should map those to the flags defined as best
	//       they are able.  Other systems may have only more broad access
	//       controls.  They will generally have just a few possible
	//       permutations of permission flags, however they should attempt to
	//       correctly represent what is permitted.
	// --------------------------------------------------------------------------------
	// 
	// PermType: unix:
	// 
	// A Unix/Linux permissions string is returned ( such as "drwxr-xr-x" or
	// "-rw-r--r--")
	//     The UNIX permissions string is 10 characters. Each character has a specific meaning. If the first character is:
	//     d 	the entry is a directory.
	//     b 	the entry is a block special file.
	//     c 	the entry is a character special file.
	//     l 	the entry is a symbolic link. Either the -N flag was specified, or the symbolic link did not point to an existing file.
	//     p 	the entry is a first-in, first-out (FIFO) special file.
	//     s 	the entry is a local socket.
	//     - 	the entry is an ordinary file.
	// 
	//     The next nine characters are divided into three sets of three characters each. The first set of three characters show 
	// the owner's permission. The next set of three characters show the permission of the other users in the group. The last
	// set of three characters shows the permission of anyone else with access to the file. The three characters in each set 
	// indicate, respectively, read, write, and execute permission of the file. With execute permission of a directory, you can search 
	// a directory for a specified file. Permissions are indicated like this:
	// 
	//     r 	read
	//     w 	write (edit)
	//     x 	execute (search)
	//     - 	corresponding permission not granted 
	// --------------------------------------------------------------------------------
	// 
	// PermType: netware:
	// 
	// Contains the NetWare rights string from a NetWare FTP server directory listing
	// format. For example "-WCE---S" or "RWCEAFMS".
	// Directory Rights	Description
	// ----------------	-------------------------------
	// Read (R)		Read data from an existing file.
	// Write (W)		Write data to an existing file.
	// Create (C)		Create a new file or subdirectory.
	// Erase (E)		Delete an existing files or directory.
	// Modify (M)	Rename and change attributes of a file.
	// File Scan (F)	List the contents of a directory.
	// Access Control (A)	Control the rights of other users to access files or directories.
	// Supervisor (S)	Automatically allowed all rights.
	// --------------------------------------------------------------------------------
	// 
	// PermType: openvms:
	// 
	// Contains the OpenVMS permissions string. For example "(RWED,RWED,RWED,RWED)",
	// "(RWED,RWED,,)", "(RWED,RWED,R,R)", etc.
	// --------------------------------------------------------------------------------
	// 
	// PermType: batchStatusFlags:
	// 
	// Contains the batch status flags from a Connect:Enterprise Server. Such as
	// "-CR--M----" or "-ART------".
	// The Batch Status Flags  is a 10-character string where each character describes an attribute of the batch. 
	// A dash indicates that flag is turned off and therefore has no meaning to the 
	// batch in question. The flags are always displayed in the same order: 
	// 
	// 1) I  -- Incomplete batch which will NOT be processed. 
	// 2) A or C -- Added or Collected
	// 3) R -- Requestable by partner 
	// 4) T -- Transmitted to partner 
	// 5) E -- Extracted (inbound file processed by McLane) 
	// 6) M -- Multi-transmittable 
	// 7) U -- Un-extractable 
	// 8) N -- Non-transmittable 
	// 9) P -- In Progress 
	// 10) - -- Always a dash.
	// 
	bool GetPermissions(int index, CkString &outStr);

	// Returns permissions information, if available, for the Nth file. If empty, then
	// no permissions information is available. The value returned by the GetPermType
	// method defines the content and format of the permissions string returned by this
	// method. Possible permission types are "mlsd", "unix", "netware", "openvms", and
	// "batchStatusFlags". The format of each permission type is as follows:
	// --------------------------------------------------------------------------------
	// 
	// PermType: mlsd:
	// 
	// A "perm fact" is returned. The format of the perm fact is defined in RFC 3659 as
	// follows:
	//   The perm fact is used to indicate access rights the current FTP user
	//    has over the object listed.  Its value is always an unordered
	//    sequence of alphabetic characters.
	// 
	//       perm-fact    = "Perm" "=" *pvals
	//       pvals        = "a" / "c" / "d" / "e" / "f" /
	//                      "l" / "m" / "p" / "r" / "w"
	// 
	//    There are ten permission indicators currently defined.  Many are
	//    meaningful only when used with a particular type of object.  The
	//    indicators are case independent, "d" and "D" are the same indicator.
	// 
	//    The "a" permission applies to objects of type=file, and indicates
	//    that the APPE (append) command may be applied to the file named.
	// 
	//    The "c" permission applies to objects of type=dir (and type=pdir,
	//    type=cdir).  It indicates that files may be created in the directory
	//    named.  That is, that a STOU command is likely to succeed, and that
	//    STOR and APPE commands might succeed if the file named did not
	//    previously exist, but is to be created in the directory object that
	//    has the "c" permission.  It also indicates that the RNTO command is
	//    likely to succeed for names in the directory.
	// 
	//    The "d" permission applies to all types.  It indicates that the
	//    object named may be deleted, that is, that the RMD command may be
	//    applied to it if it is a directory, and otherwise that the DELE
	//    command may be applied to it.
	// 
	//    The "e" permission applies to the directory types.  When set on an
	//    object of type=dir, type=cdir, or type=pdir it indicates that a CWD
	//    command naming the object should succeed, and the user should be able
	//    to enter the directory named.  For type=pdir it also indicates that
	//    the CDUP command may succeed (if this particular pathname is the one
	//    to which a CDUP would apply.)
	// 
	//    The "f" permission for objects indicates that the object named may be
	//    renamed - that is, may be the object of an RNFR command.
	// 
	//    The "l" permission applies to the directory file types, and indicates
	//    that the listing commands, LIST, NLST, and MLSD may be applied to the
	//    directory in question.
	// 
	//    The "m" permission applies to directory types, and indicates that the
	//    MKD command may be used to create a new directory within the
	//    directory under consideration.
	// 
	//    The "p" permission applies to directory types, and indicates that
	//    objects in the directory may be deleted, or (stretching naming a
	//    little) that the directory may be purged.  Note: it does not indicate
	//    that the RMD command may be used to remove the directory named
	//    itself, the "d" permission indicator indicates that.
	// 
	//    The "r" permission applies to type=file objects, and for some
	//    systems, perhaps to other types of objects, and indicates that the
	//    RETR command may be applied to that object.
	// 
	//    The "w" permission applies to type=file objects, and for some
	//    systems, perhaps to other types of objects, and indicates that the
	//    STOR command may be applied to the object named.
	// 
	//    Note: That a permission indicator is set can never imply that the
	//       appropriate command is guaranteed to work -- just that it might.
	//       Other system specific limitations, such as limitations on
	//       available space for storing files, may cause an operation to fail,
	//       where the permission flags may have indicated that it was likely
	//       to succeed.  The permissions are a guide only.
	// 
	//    Implementation note: The permissions are described here as they apply
	//       to FTP commands.  They may not map easily into particular
	//       permissions available on the server's operating system.  Servers
	//       are expected to synthesize these permission bits from the
	//       permission information available from operating system.  For
	//       example, to correctly determine whether the "D" permission bit
	//       should be set on a directory for a server running on the UNIX(TM)
	//       operating system, the server should check that the directory named
	//       is empty, and that the user has write permission on both the
	//       directory under consideration, and its parent directory.
	// 
	//       Some systems may have more specific permissions than those listed
	//       here, such systems should map those to the flags defined as best
	//       they are able.  Other systems may have only more broad access
	//       controls.  They will generally have just a few possible
	//       permutations of permission flags, however they should attempt to
	//       correctly represent what is permitted.
	// --------------------------------------------------------------------------------
	// 
	// PermType: unix:
	// 
	// A Unix/Linux permissions string is returned ( such as "drwxr-xr-x" or
	// "-rw-r--r--")
	//     The UNIX permissions string is 10 characters. Each character has a specific meaning. If the first character is:
	//     d 	the entry is a directory.
	//     b 	the entry is a block special file.
	//     c 	the entry is a character special file.
	//     l 	the entry is a symbolic link. Either the -N flag was specified, or the symbolic link did not point to an existing file.
	//     p 	the entry is a first-in, first-out (FIFO) special file.
	//     s 	the entry is a local socket.
	//     - 	the entry is an ordinary file.
	// 
	//     The next nine characters are divided into three sets of three characters each. The first set of three characters show 
	// the owner's permission. The next set of three characters show the permission of the other users in the group. The last
	// set of three characters shows the permission of anyone else with access to the file. The three characters in each set 
	// indicate, respectively, read, write, and execute permission of the file. With execute permission of a directory, you can search 
	// a directory for a specified file. Permissions are indicated like this:
	// 
	//     r 	read
	//     w 	write (edit)
	//     x 	execute (search)
	//     - 	corresponding permission not granted 
	// --------------------------------------------------------------------------------
	// 
	// PermType: netware:
	// 
	// Contains the NetWare rights string from a NetWare FTP server directory listing
	// format. For example "-WCE---S" or "RWCEAFMS".
	// Directory Rights	Description
	// ----------------	-------------------------------
	// Read (R)		Read data from an existing file.
	// Write (W)		Write data to an existing file.
	// Create (C)		Create a new file or subdirectory.
	// Erase (E)		Delete an existing files or directory.
	// Modify (M)	Rename and change attributes of a file.
	// File Scan (F)	List the contents of a directory.
	// Access Control (A)	Control the rights of other users to access files or directories.
	// Supervisor (S)	Automatically allowed all rights.
	// --------------------------------------------------------------------------------
	// 
	// PermType: openvms:
	// 
	// Contains the OpenVMS permissions string. For example "(RWED,RWED,RWED,RWED)",
	// "(RWED,RWED,,)", "(RWED,RWED,R,R)", etc.
	// --------------------------------------------------------------------------------
	// 
	// PermType: batchStatusFlags:
	// 
	// Contains the batch status flags from a Connect:Enterprise Server. Such as
	// "-CR--M----" or "-ART------".
	// The Batch Status Flags  is a 10-character string where each character describes an attribute of the batch. 
	// A dash indicates that flag is turned off and therefore has no meaning to the 
	// batch in question. The flags are always displayed in the same order: 
	// 
	// 1) I  -- Incomplete batch which will NOT be processed. 
	// 2) A or C -- Added or Collected
	// 3) R -- Requestable by partner 
	// 4) T -- Transmitted to partner 
	// 5) E -- Extracted (inbound file processed by McLane) 
	// 6) M -- Multi-transmittable 
	// 7) U -- Un-extractable 
	// 8) N -- Non-transmittable 
	// 9) P -- In Progress 
	// 10) - -- Always a dash.
	// 
	const char *getPermissions(int index);
	// Returns permissions information, if available, for the Nth file. If empty, then
	// no permissions information is available. The value returned by the GetPermType
	// method defines the content and format of the permissions string returned by this
	// method. Possible permission types are "mlsd", "unix", "netware", "openvms", and
	// "batchStatusFlags". The format of each permission type is as follows:
	// --------------------------------------------------------------------------------
	// 
	// PermType: mlsd:
	// 
	// A "perm fact" is returned. The format of the perm fact is defined in RFC 3659 as
	// follows:
	//   The perm fact is used to indicate access rights the current FTP user
	//    has over the object listed.  Its value is always an unordered
	//    sequence of alphabetic characters.
	// 
	//       perm-fact    = "Perm" "=" *pvals
	//       pvals        = "a" / "c" / "d" / "e" / "f" /
	//                      "l" / "m" / "p" / "r" / "w"
	// 
	//    There are ten permission indicators currently defined.  Many are
	//    meaningful only when used with a particular type of object.  The
	//    indicators are case independent, "d" and "D" are the same indicator.
	// 
	//    The "a" permission applies to objects of type=file, and indicates
	//    that the APPE (append) command may be applied to the file named.
	// 
	//    The "c" permission applies to objects of type=dir (and type=pdir,
	//    type=cdir).  It indicates that files may be created in the directory
	//    named.  That is, that a STOU command is likely to succeed, and that
	//    STOR and APPE commands might succeed if the file named did not
	//    previously exist, but is to be created in the directory object that
	//    has the "c" permission.  It also indicates that the RNTO command is
	//    likely to succeed for names in the directory.
	// 
	//    The "d" permission applies to all types.  It indicates that the
	//    object named may be deleted, that is, that the RMD command may be
	//    applied to it if it is a directory, and otherwise that the DELE
	//    command may be applied to it.
	// 
	//    The "e" permission applies to the directory types.  When set on an
	//    object of type=dir, type=cdir, or type=pdir it indicates that a CWD
	//    command naming the object should succeed, and the user should be able
	//    to enter the directory named.  For type=pdir it also indicates that
	//    the CDUP command may succeed (if this particular pathname is the one
	//    to which a CDUP would apply.)
	// 
	//    The "f" permission for objects indicates that the object named may be
	//    renamed - that is, may be the object of an RNFR command.
	// 
	//    The "l" permission applies to the directory file types, and indicates
	//    that the listing commands, LIST, NLST, and MLSD may be applied to the
	//    directory in question.
	// 
	//    The "m" permission applies to directory types, and indicates that the
	//    MKD command may be used to create a new directory within the
	//    directory under consideration.
	// 
	//    The "p" permission applies to directory types, and indicates that
	//    objects in the directory may be deleted, or (stretching naming a
	//    little) that the directory may be purged.  Note: it does not indicate
	//    that the RMD command may be used to remove the directory named
	//    itself, the "d" permission indicator indicates that.
	// 
	//    The "r" permission applies to type=file objects, and for some
	//    systems, perhaps to other types of objects, and indicates that the
	//    RETR command may be applied to that object.
	// 
	//    The "w" permission applies to type=file objects, and for some
	//    systems, perhaps to other types of objects, and indicates that the
	//    STOR command may be applied to the object named.
	// 
	//    Note: That a permission indicator is set can never imply that the
	//       appropriate command is guaranteed to work -- just that it might.
	//       Other system specific limitations, such as limitations on
	//       available space for storing files, may cause an operation to fail,
	//       where the permission flags may have indicated that it was likely
	//       to succeed.  The permissions are a guide only.
	// 
	//    Implementation note: The permissions are described here as they apply
	//       to FTP commands.  They may not map easily into particular
	//       permissions available on the server's operating system.  Servers
	//       are expected to synthesize these permission bits from the
	//       permission information available from operating system.  For
	//       example, to correctly determine whether the "D" permission bit
	//       should be set on a directory for a server running on the UNIX(TM)
	//       operating system, the server should check that the directory named
	//       is empty, and that the user has write permission on both the
	//       directory under consideration, and its parent directory.
	// 
	//       Some systems may have more specific permissions than those listed
	//       here, such systems should map those to the flags defined as best
	//       they are able.  Other systems may have only more broad access
	//       controls.  They will generally have just a few possible
	//       permutations of permission flags, however they should attempt to
	//       correctly represent what is permitted.
	// --------------------------------------------------------------------------------
	// 
	// PermType: unix:
	// 
	// A Unix/Linux permissions string is returned ( such as "drwxr-xr-x" or
	// "-rw-r--r--")
	//     The UNIX permissions string is 10 characters. Each character has a specific meaning. If the first character is:
	//     d 	the entry is a directory.
	//     b 	the entry is a block special file.
	//     c 	the entry is a character special file.
	//     l 	the entry is a symbolic link. Either the -N flag was specified, or the symbolic link did not point to an existing file.
	//     p 	the entry is a first-in, first-out (FIFO) special file.
	//     s 	the entry is a local socket.
	//     - 	the entry is an ordinary file.
	// 
	//     The next nine characters are divided into three sets of three characters each. The first set of three characters show 
	// the owner's permission. The next set of three characters show the permission of the other users in the group. The last
	// set of three characters shows the permission of anyone else with access to the file. The three characters in each set 
	// indicate, respectively, read, write, and execute permission of the file. With execute permission of a directory, you can search 
	// a directory for a specified file. Permissions are indicated like this:
	// 
	//     r 	read
	//     w 	write (edit)
	//     x 	execute (search)
	//     - 	corresponding permission not granted 
	// --------------------------------------------------------------------------------
	// 
	// PermType: netware:
	// 
	// Contains the NetWare rights string from a NetWare FTP server directory listing
	// format. For example "-WCE---S" or "RWCEAFMS".
	// Directory Rights	Description
	// ----------------	-------------------------------
	// Read (R)		Read data from an existing file.
	// Write (W)		Write data to an existing file.
	// Create (C)		Create a new file or subdirectory.
	// Erase (E)		Delete an existing files or directory.
	// Modify (M)	Rename and change attributes of a file.
	// File Scan (F)	List the contents of a directory.
	// Access Control (A)	Control the rights of other users to access files or directories.
	// Supervisor (S)	Automatically allowed all rights.
	// --------------------------------------------------------------------------------
	// 
	// PermType: openvms:
	// 
	// Contains the OpenVMS permissions string. For example "(RWED,RWED,RWED,RWED)",
	// "(RWED,RWED,,)", "(RWED,RWED,R,R)", etc.
	// --------------------------------------------------------------------------------
	// 
	// PermType: batchStatusFlags:
	// 
	// Contains the batch status flags from a Connect:Enterprise Server. Such as
	// "-CR--M----" or "-ART------".
	// The Batch Status Flags  is a 10-character string where each character describes an attribute of the batch. 
	// A dash indicates that flag is turned off and therefore has no meaning to the 
	// batch in question. The flags are always displayed in the same order: 
	// 
	// 1) I  -- Incomplete batch which will NOT be processed. 
	// 2) A or C -- Added or Collected
	// 3) R -- Requestable by partner 
	// 4) T -- Transmitted to partner 
	// 5) E -- Extracted (inbound file processed by McLane) 
	// 6) M -- Multi-transmittable 
	// 7) U -- Un-extractable 
	// 8) N -- Non-transmittable 
	// 9) P -- In Progress 
	// 10) - -- Always a dash.
	// 
	const char *permissions(int index);

	// Returns permissions information, if available, for the Nth file. If empty, then
	// no permissions information is available. The value returned by the GetPermType
	// method defines the content and format of the permissions string returned by this
	// method. Possible permission types are "mlsd", "unix", "netware", "openvms", and
	// "batchStatusFlags". The format of each permission type is as follows:
	// --------------------------------------------------------------------------------
	// 
	// PermType: mlsd:
	// 
	// A "perm fact" is returned. The format of the perm fact is defined in RFC 3659 as
	// follows:
	//   The perm fact is used to indicate access rights the current FTP user
	//    has over the object listed.  Its value is always an unordered
	//    sequence of alphabetic characters.
	// 
	//       perm-fact    = "Perm" "=" *pvals
	//       pvals        = "a" / "c" / "d" / "e" / "f" /
	//                      "l" / "m" / "p" / "r" / "w"
	// 
	//    There are ten permission indicators currently defined.  Many are
	//    meaningful only when used with a particular type of object.  The
	//    indicators are case independent, "d" and "D" are the same indicator.
	// 
	//    The "a" permission applies to objects of type=file, and indicates
	//    that the APPE (append) command may be applied to the file named.
	// 
	//    The "c" permission applies to objects of type=dir (and type=pdir,
	//    type=cdir).  It indicates that files may be created in the directory
	//    named.  That is, that a STOU command is likely to succeed, and that
	//    STOR and APPE commands might succeed if the file named did not
	//    previously exist, but is to be created in the directory object that
	//    has the "c" permission.  It also indicates that the RNTO command is
	//    likely to succeed for names in the directory.
	// 
	//    The "d" permission applies to all types.  It indicates that the
	//    object named may be deleted, that is, that the RMD command may be
	//    applied to it if it is a directory, and otherwise that the DELE
	//    command may be applied to it.
	// 
	//    The "e" permission applies to the directory types.  When set on an
	//    object of type=dir, type=cdir, or type=pdir it indicates that a CWD
	//    command naming the object should succeed, and the user should be able
	//    to enter the directory named.  For type=pdir it also indicates that
	//    the CDUP command may succeed (if this particular pathname is the one
	//    to which a CDUP would apply.)
	// 
	//    The "f" permission for objects indicates that the object named may be
	//    renamed - that is, may be the object of an RNFR command.
	// 
	//    The "l" permission applies to the directory file types, and indicates
	//    that the listing commands, LIST, NLST, and MLSD may be applied to the
	//    directory in question.
	// 
	//    The "m" permission applies to directory types, and indicates that the
	//    MKD command may be used to create a new directory within the
	//    directory under consideration.
	// 
	//    The "p" permission applies to directory types, and indicates that
	//    objects in the directory may be deleted, or (stretching naming a
	//    little) that the directory may be purged.  Note: it does not indicate
	//    that the RMD command may be used to remove the directory named
	//    itself, the "d" permission indicator indicates that.
	// 
	//    The "r" permission applies to type=file objects, and for some
	//    systems, perhaps to other types of objects, and indicates that the
	//    RETR command may be applied to that object.
	// 
	//    The "w" permission applies to type=file objects, and for some
	//    systems, perhaps to other types of objects, and indicates that the
	//    STOR command may be applied to the object named.
	// 
	//    Note: That a permission indicator is set can never imply that the
	//       appropriate command is guaranteed to work -- just that it might.
	//       Other system specific limitations, such as limitations on
	//       available space for storing files, may cause an operation to fail,
	//       where the permission flags may have indicated that it was likely
	//       to succeed.  The permissions are a guide only.
	// 
	//    Implementation note: The permissions are described here as they apply
	//       to FTP commands.  They may not map easily into particular
	//       permissions available on the server's operating system.  Servers
	//       are expected to synthesize these permission bits from the
	//       permission information available from operating system.  For
	//       example, to correctly determine whether the "D" permission bit
	//       should be set on a directory for a server running on the UNIX(TM)
	//       operating system, the server should check that the directory named
	//       is empty, and that the user has write permission on both the
	//       directory under consideration, and its parent directory.
	// 
	//       Some systems may have more specific permissions than those listed
	//       here, such systems should map those to the flags defined as best
	//       they are able.  Other systems may have only more broad access
	//       controls.  They will generally have just a few possible
	//       permutations of permission flags, however they should attempt to
	//       correctly represent what is permitted.
	// --------------------------------------------------------------------------------
	// 
	// PermType: unix:
	// 
	// A Unix/Linux permissions string is returned ( such as "drwxr-xr-x" or
	// "-rw-r--r--")
	//     The UNIX permissions string is 10 characters. Each character has a specific meaning. If the first character is:
	//     d 	the entry is a directory.
	//     b 	the entry is a block special file.
	//     c 	the entry is a character special file.
	//     l 	the entry is a symbolic link. Either the -N flag was specified, or the symbolic link did not point to an existing file.
	//     p 	the entry is a first-in, first-out (FIFO) special file.
	//     s 	the entry is a local socket.
	//     - 	the entry is an ordinary file.
	// 
	//     The next nine characters are divided into three sets of three characters each. The first set of three characters show 
	// the owner's permission. The next set of three characters show the permission of the other users in the group. The last
	// set of three characters shows the permission of anyone else with access to the file. The three characters in each set 
	// indicate, respectively, read, write, and execute permission of the file. With execute permission of a directory, you can search 
	// a directory for a specified file. Permissions are indicated like this:
	// 
	//     r 	read
	//     w 	write (edit)
	//     x 	execute (search)
	//     - 	corresponding permission not granted 
	// --------------------------------------------------------------------------------
	// 
	// PermType: netware:
	// 
	// Contains the NetWare rights string from a NetWare FTP server directory listing
	// format. For example "-WCE---S" or "RWCEAFMS".
	// Directory Rights	Description
	// ----------------	-------------------------------
	// Read (R)		Read data from an existing file.
	// Write (W)		Write data to an existing file.
	// Create (C)		Create a new file or subdirectory.
	// Erase (E)		Delete an existing files or directory.
	// Modify (M)	Rename and change attributes of a file.
	// File Scan (F)	List the contents of a directory.
	// Access Control (A)	Control the rights of other users to access files or directories.
	// Supervisor (S)	Automatically allowed all rights.
	// --------------------------------------------------------------------------------
	// 
	// PermType: openvms:
	// 
	// Contains the OpenVMS permissions string. For example "(RWED,RWED,RWED,RWED)",
	// "(RWED,RWED,,)", "(RWED,RWED,R,R)", etc.
	// --------------------------------------------------------------------------------
	// 
	// PermType: batchStatusFlags:
	// 
	// Contains the batch status flags from a Connect:Enterprise Server. Such as
	// "-CR--M----" or "-ART------".
	// The Batch Status Flags  is a 10-character string where each character describes an attribute of the batch. 
	// A dash indicates that flag is turned off and therefore has no meaning to the 
	// batch in question. The flags are always displayed in the same order: 
	// 
	// 1) I  -- Incomplete batch which will NOT be processed. 
	// 2) A or C -- Added or Collected
	// 3) R -- Requestable by partner 
	// 4) T -- Transmitted to partner 
	// 5) E -- Extracted (inbound file processed by McLane) 
	// 6) M -- Multi-transmittable 
	// 7) U -- Un-extractable 
	// 8) N -- Non-transmittable 
	// 9) P -- In Progress 
	// 10) - -- Always a dash.
	// 
	CkTask *GetPermissionsAsync(int index);


	// Returns the type of permissions information that is available for the Nth file.
	// If empty, then no permissions information is available. The value returned by
	// this method defines the content and format of the permissions string returned by
	// the GetPermissions method. Possible values are "mlsd", "unix", "netware",
	// "openvms", and "batchStatusFlags".
	bool GetPermType(int index, CkString &outStr);

	// Returns the type of permissions information that is available for the Nth file.
	// If empty, then no permissions information is available. The value returned by
	// this method defines the content and format of the permissions string returned by
	// the GetPermissions method. Possible values are "mlsd", "unix", "netware",
	// "openvms", and "batchStatusFlags".
	const char *getPermType(int index);
	// Returns the type of permissions information that is available for the Nth file.
	// If empty, then no permissions information is available. The value returned by
	// this method defines the content and format of the permissions string returned by
	// the GetPermissions method. Possible values are "mlsd", "unix", "netware",
	// "openvms", and "batchStatusFlags".
	const char *permType(int index);

	// Returns the type of permissions information that is available for the Nth file.
	// If empty, then no permissions information is available. The value returned by
	// this method defines the content and format of the permissions string returned by
	// the GetPermissions method. Possible values are "mlsd", "unix", "netware",
	// "openvms", and "batchStatusFlags".
	CkTask *GetPermTypeAsync(int index);


	// Downloads the contents of a remote file into a byte array.
	bool GetRemoteFileBinaryData(const char *remoteFilename, CkByteData &outData);

	// Downloads the contents of a remote file into a byte array.
	CkTask *GetRemoteFileBinaryDataAsync(const char *remoteFilename);


	// Downloads a text file directly into a string variable. The character encoding of
	// the text file is specified by the charset argument, which is a value such as utf-8,
	// iso-8859-1, Shift_JIS, etc.
	bool GetRemoteFileTextC(const char *remoteFilename, const char *charset, CkString &outStr);

	// Downloads a text file directly into a string variable. The character encoding of
	// the text file is specified by the charset argument, which is a value such as utf-8,
	// iso-8859-1, Shift_JIS, etc.
	const char *getRemoteFileTextC(const char *remoteFilename, const char *charset);
	// Downloads a text file directly into a string variable. The character encoding of
	// the text file is specified by the charset argument, which is a value such as utf-8,
	// iso-8859-1, Shift_JIS, etc.
	const char *remoteFileTextC(const char *remoteFilename, const char *charset);

	// Downloads a text file directly into a string variable. The character encoding of
	// the text file is specified by the charset argument, which is a value such as utf-8,
	// iso-8859-1, Shift_JIS, etc.
	CkTask *GetRemoteFileTextCAsync(const char *remoteFilename, const char *charset);


	// Downloads the content of a remote text file directly into an in-memory string.
	// 
	// Note: If the remote text file does not use the ANSI character encoding, call
	// GetRemoteFileTextC instead, which allows for the character encoding to be
	// specified so that characters are properly interpreted.
	// 
	bool GetRemoteFileTextData(const char *remoteFilename, CkString &outStr);

	// Downloads the content of a remote text file directly into an in-memory string.
	// 
	// Note: If the remote text file does not use the ANSI character encoding, call
	// GetRemoteFileTextC instead, which allows for the character encoding to be
	// specified so that characters are properly interpreted.
	// 
	const char *getRemoteFileTextData(const char *remoteFilename);
	// Downloads the content of a remote text file directly into an in-memory string.
	// 
	// Note: If the remote text file does not use the ANSI character encoding, call
	// GetRemoteFileTextC instead, which allows for the character encoding to be
	// specified so that characters are properly interpreted.
	// 
	const char *remoteFileTextData(const char *remoteFilename);

	// Downloads the content of a remote text file directly into an in-memory string.
	// 
	// Note: If the remote text file does not use the ANSI character encoding, call
	// GetRemoteFileTextC instead, which allows for the character encoding to be
	// specified so that characters are properly interpreted.
	// 
	CkTask *GetRemoteFileTextDataAsync(const char *remoteFilename);


	// Returns the size of the Nth remote file in the current directory.
	int GetSize(int index);

	// Returns the size of the Nth remote file in the current directory.
	CkTask *GetSizeAsync(int index);


	// Returns the size of the Nth remote file in the current directory as a 64-bit
	// integer. Returns -1 if the file does not exist.
	__int64 GetSize64(int index);


	// Returns a remote file's size in bytes. Returns -1 if the file does not exist.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	int GetSizeByName(const char *filename);

	// Returns a remote file's size in bytes. Returns -1 if the file does not exist.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	CkTask *GetSizeByNameAsync(const char *filename);


	// Returns a remote file's size in bytes as a 64-bit integer.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	__int64 GetSizeByName64(const char *filename);


	// Returns the size in decimal string format of the Nth remote file in the current
	// directory. This is helpful for cases when the file size (in bytes) is greater
	// than what can fit in a 32-bit integer.
	bool GetSizeStr(int index, CkString &outStr);

	// Returns the size in decimal string format of the Nth remote file in the current
	// directory. This is helpful for cases when the file size (in bytes) is greater
	// than what can fit in a 32-bit integer.
	const char *getSizeStr(int index);
	// Returns the size in decimal string format of the Nth remote file in the current
	// directory. This is helpful for cases when the file size (in bytes) is greater
	// than what can fit in a 32-bit integer.
	const char *sizeStr(int index);

	// Returns the size in decimal string format of the Nth remote file in the current
	// directory. This is helpful for cases when the file size (in bytes) is greater
	// than what can fit in a 32-bit integer.
	CkTask *GetSizeStrAsync(int index);


	// Returns the size of a remote file as a string. This is helpful when file a file
	// size is greater than what can fit in a 32-bit integer.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	bool GetSizeStrByName(const char *filename, CkString &outStr);

	// Returns the size of a remote file as a string. This is helpful when file a file
	// size is greater than what can fit in a 32-bit integer.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	const char *getSizeStrByName(const char *filename);
	// Returns the size of a remote file as a string. This is helpful when file a file
	// size is greater than what can fit in a 32-bit integer.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	const char *sizeStrByName(const char *filename);

	// Returns the size of a remote file as a string. This is helpful when file a file
	// size is greater than what can fit in a 32-bit integer.
	// 
	// Note: The filename passed to this method must NOT include a path. Prior to calling
	// this method, make sure to set the current remote directory (via the
	// ChangeRemoteDir method) to the remote directory where this file exists.
	// 
	// Note: Prior to calling this method, it should be ensured that the ListPattern
	// property is set to a pattern that would match the requested filename. (The default
	// value of ListPattern is "*", which will match all filenames.)
	// 
	CkTask *GetSizeStrByNameAsync(const char *filename);


	// Returns the FTP server's digital certificate (for SSL / TLS connections).
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetSslServerCert(void);


	// Returns a listing of the files and directories in the current directory matching
	// the pattern. Passing "*.*" will return all the files and directories.
	bool GetTextDirListing(const char *pattern, CkString &outStrRawListing);

	// Returns a listing of the files and directories in the current directory matching
	// the pattern. Passing "*.*" will return all the files and directories.
	const char *getTextDirListing(const char *pattern);
	// Returns a listing of the files and directories in the current directory matching
	// the pattern. Passing "*.*" will return all the files and directories.
	const char *textDirListing(const char *pattern);

	// Returns a listing of the files and directories in the current directory matching
	// the pattern. Passing "*.*" will return all the files and directories.
	CkTask *GetTextDirListingAsync(const char *pattern);


	// Returns (in XML format) the files and directories in the current directory
	// matching the pattern. Passing "*.*" will return all the files and directories.
	// 
	// Note: The lastModTime XML elements contain date/time information in the local
	// (client) timezone. However, it's possible based on the capabilities of an FTP
	// server (or lack of capabilities) that the timezone information for the remote
	// files is not available. In other words, in some cases, the timezone of an FTP
	// server cannot be known, especially for older FTP server implementations.
	// 
	bool GetXmlDirListing(const char *pattern, CkString &outStrXmlListing);

	// Returns (in XML format) the files and directories in the current directory
	// matching the pattern. Passing "*.*" will return all the files and directories.
	// 
	// Note: The lastModTime XML elements contain date/time information in the local
	// (client) timezone. However, it's possible based on the capabilities of an FTP
	// server (or lack of capabilities) that the timezone information for the remote
	// files is not available. In other words, in some cases, the timezone of an FTP
	// server cannot be known, especially for older FTP server implementations.
	// 
	const char *getXmlDirListing(const char *pattern);
	// Returns (in XML format) the files and directories in the current directory
	// matching the pattern. Passing "*.*" will return all the files and directories.
	// 
	// Note: The lastModTime XML elements contain date/time information in the local
	// (client) timezone. However, it's possible based on the capabilities of an FTP
	// server (or lack of capabilities) that the timezone information for the remote
	// files is not available. In other words, in some cases, the timezone of an FTP
	// server cannot be known, especially for older FTP server implementations.
	// 
	const char *xmlDirListing(const char *pattern);

	// Returns (in XML format) the files and directories in the current directory
	// matching the pattern. Passing "*.*" will return all the files and directories.
	// 
	// Note: The lastModTime XML elements contain date/time information in the local
	// (client) timezone. However, it's possible based on the capabilities of an FTP
	// server (or lack of capabilities) that the timezone information for the remote
	// files is not available. In other words, in some cases, the timezone of an FTP
	// server cannot be known, especially for older FTP server implementations.
	// 
	CkTask *GetXmlDirListingAsync(const char *pattern);


	// Return true if the component is already unlocked.
	bool IsUnlocked(void);


	// This is the same as PutFile, but designed to work around the following potential
	// problem associated with an upload that is extremely large.
	// 
	// FTP uses two TCP (or TLS) connections: a control connection to submit commands
	// and receive replies, and a data connection for actual file transfers. It is the
	// nature of FTP that during a transfer the control connection stays completely
	// idle. Many routers and firewalls automatically close idle connections after a
	// certain period of time. Worse, they often don't notify the user, but just
	// silently drop the connection.
	// 
	// For FTP, this means that during a long transfer the control connection can get
	// dropped because it is detected as idle, but neither client nor server are
	// notified. When all data has been transferred, the server assumes the control
	// connection is alive and it sends the transfer confirmation reply.
	// 
	// Likewise, the client thinks the control connection is alive and it waits for the
	// reply from the server. But since the control connection got dropped without
	// notification, the reply never arrives and eventually the connection will
	// timeout.
	// 
	// The Solution: This method uploads the file in chunks, where each chunk appends
	// to the remote file. This way, each chunk is a separate FTP upload that does not
	// take too long to complete. The chunkSize specifies the number of bytes to upload in
	// each chunk. The size should be based on the amount of memory available (because
	// each chunk will reside in memory as it's being uploaded), the transfer rate, and
	// the total size of the file being uploaded. For example, if a 4GB file is
	// uploaded, and the chunkSize is set to 1MB (1,048,576 bytes), then 4000 separate
	// chunks would be required. This is likely not a good choice for chunkSize. A more
	// appropriate chunkSize might be 20MB, in which case the upload would complete in 200
	// separate chunks. The application would temporarily be using a 20MB buffer for
	// uploading chunks. The tradeoff is between the number of chunks (the more chunks,
	// the larger the overall time to upload), the amount of memory that is reasonable
	// for the temporary buffer, and the amount of time required to upload each chunk
	// (if the chunk size is too large, then the problem described above is not
	// solved).
	// 
	bool LargeFileUpload(const char *localPath, const char *remotePath, int chunkSize);

	// This is the same as PutFile, but designed to work around the following potential
	// problem associated with an upload that is extremely large.
	// 
	// FTP uses two TCP (or TLS) connections: a control connection to submit commands
	// and receive replies, and a data connection for actual file transfers. It is the
	// nature of FTP that during a transfer the control connection stays completely
	// idle. Many routers and firewalls automatically close idle connections after a
	// certain period of time. Worse, they often don't notify the user, but just
	// silently drop the connection.
	// 
	// For FTP, this means that during a long transfer the control connection can get
	// dropped because it is detected as idle, but neither client nor server are
	// notified. When all data has been transferred, the server assumes the control
	// connection is alive and it sends the transfer confirmation reply.
	// 
	// Likewise, the client thinks the control connection is alive and it waits for the
	// reply from the server. But since the control connection got dropped without
	// notification, the reply never arrives and eventually the connection will
	// timeout.
	// 
	// The Solution: This method uploads the file in chunks, where each chunk appends
	// to the remote file. This way, each chunk is a separate FTP upload that does not
	// take too long to complete. The chunkSize specifies the number of bytes to upload in
	// each chunk. The size should be based on the amount of memory available (because
	// each chunk will reside in memory as it's being uploaded), the transfer rate, and
	// the total size of the file being uploaded. For example, if a 4GB file is
	// uploaded, and the chunkSize is set to 1MB (1,048,576 bytes), then 4000 separate
	// chunks would be required. This is likely not a good choice for chunkSize. A more
	// appropriate chunkSize might be 20MB, in which case the upload would complete in 200
	// separate chunks. The application would temporarily be using a 20MB buffer for
	// uploading chunks. The tradeoff is between the number of chunks (the more chunks,
	// the larger the overall time to upload), the amount of memory that is reasonable
	// for the temporary buffer, and the amount of time required to upload each chunk
	// (if the chunk size is too large, then the problem described above is not
	// solved).
	// 
	CkTask *LargeFileUploadAsync(const char *localPath, const char *remotePath, int chunkSize);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Authenticates with the FTP server using the values provided in the Username,
	// Password, and/or other properties. This can be called after establishing the
	// connection via the ConnectOnly method. (The Connect method both connects and
	// authenticates.) The combination of calling ConnectOnly followed by
	// LoginAfterConnectOnly is the equivalent of calling the Connect method.
	// 
	// Note: After successful authentication, the FEAT and SYST commands are
	// automatically sent to help the client understand what is supported by the FTP
	// server. To prevent these commands from being sent, set the AutoFeat and/or
	// AutoSyst properties equal to false.
	// 
	bool LoginAfterConnectOnly(void);

	// Authenticates with the FTP server using the values provided in the Username,
	// Password, and/or other properties. This can be called after establishing the
	// connection via the ConnectOnly method. (The Connect method both connects and
	// authenticates.) The combination of calling ConnectOnly followed by
	// LoginAfterConnectOnly is the equivalent of calling the Connect method.
	// 
	// Note: After successful authentication, the FEAT and SYST commands are
	// automatically sent to help the client understand what is supported by the FTP
	// server. To prevent these commands from being sent, set the AutoFeat and/or
	// AutoSyst properties equal to false.
	// 
	CkTask *LoginAfterConnectOnlyAsync(void);


	// Copies all the files in the current remote FTP directory to a local directory.
	// To copy all the files in a remote directory, set remotePattern to "*.*" The
	// pattern can contain any number of "*"characters, where "*" matches 0 or more of
	// any character. The return value is the number of files transferred, and on
	// error, a value of -1 is returned. Detailed information about the transfer can be
	// obtained from the last-error information
	// (LastErrorText/LastErrorHtml/LastErrorXml/SaveLastError).
	// 
	// About case sensitivity: The MGetFiles command works by sending the "LIST"
	// command to the FTP server. For example: "LIST *.txt". The FTP server responds
	// with a directory listing of the files matching the wildcarded pattern, and it is
	// these files that are downloaded. Case sensitivity depends on the
	// case-sensitivity of the remote file system. If the FTP server is running on a
	// Windows-based computer, it is likely to be case insensitive. However, if the FTP
	// server is running on Linux, MAC OS X, etc. it is likely to be case sensitive.
	// There is no good way to force case-insensitivity if the remote filesystem is
	// case-sensitive because it is not possible for the FTP client to send a LIST
	// command indicating that it wants the matching to be case-insensitive.
	// 
	int MGetFiles(const char *remotePattern, const char *localDir);

	// Copies all the files in the current remote FTP directory to a local directory.
	// To copy all the files in a remote directory, set remotePattern to "*.*" The
	// pattern can contain any number of "*"characters, where "*" matches 0 or more of
	// any character. The return value is the number of files transferred, and on
	// error, a value of -1 is returned. Detailed information about the transfer can be
	// obtained from the last-error information
	// (LastErrorText/LastErrorHtml/LastErrorXml/SaveLastError).
	// 
	// About case sensitivity: The MGetFiles command works by sending the "LIST"
	// command to the FTP server. For example: "LIST *.txt". The FTP server responds
	// with a directory listing of the files matching the wildcarded pattern, and it is
	// these files that are downloaded. Case sensitivity depends on the
	// case-sensitivity of the remote file system. If the FTP server is running on a
	// Windows-based computer, it is likely to be case insensitive. However, if the FTP
	// server is running on Linux, MAC OS X, etc. it is likely to be case sensitive.
	// There is no good way to force case-insensitivity if the remote filesystem is
	// case-sensitive because it is not possible for the FTP client to send a LIST
	// command indicating that it wants the matching to be case-insensitive.
	// 
	CkTask *MGetFilesAsync(const char *remotePattern, const char *localDir);


	// Uploads all the files matching pattern on the local computer to the current
	// remote FTP directory. The pattern parameter can include directory information,
	// such as "C:/my_dir/*.txt" or it can simply be a pattern such as "*.*" that
	// matches the files in the application's current directory. Subdirectories are not
	// recursed. The return value is the number of files copied, with a value of -1
	// returned for errors. Detailed information about the transfer can be obtained
	// from the XML log.
	int MPutFiles(const char *pattern);

	// Uploads all the files matching pattern on the local computer to the current
	// remote FTP directory. The pattern parameter can include directory information,
	// such as "C:/my_dir/*.txt" or it can simply be a pattern such as "*.*" that
	// matches the files in the application's current directory. Subdirectories are not
	// recursed. The return value is the number of files copied, with a value of -1
	// returned for errors. Detailed information about the transfer can be obtained
	// from the XML log.
	CkTask *MPutFilesAsync(const char *pattern);


	// Sends an NLST command to the FTP server and returns the results in XML format.
	// The NLST command returns a list of filenames in the given directory (matching
	// the pattern). The remoteDirPattern should be a pattern such as "*", "*.*", "*.txt",
	// "subDir/*.xml", etc.
	// 
	// The format of the XML returned is:
	// filename_or_dir_1filename_or_dir_2filename_or_dir_3filename_or_dir_4...
	// 
	bool NlstXml(const char *remoteDirPattern, CkString &outStr);

	// Sends an NLST command to the FTP server and returns the results in XML format.
	// The NLST command returns a list of filenames in the given directory (matching
	// the pattern). The remoteDirPattern should be a pattern such as "*", "*.*", "*.txt",
	// "subDir/*.xml", etc.
	// 
	// The format of the XML returned is:
	// filename_or_dir_1filename_or_dir_2filename_or_dir_3filename_or_dir_4...
	// 
	const char *nlstXml(const char *remoteDirPattern);
	// Sends an NLST command to the FTP server and returns the results in XML format.
	// The NLST command returns a list of filenames in the given directory (matching
	// the pattern). The remoteDirPattern should be a pattern such as "*", "*.*", "*.txt",
	// "subDir/*.xml", etc.
	// 
	// The format of the XML returned is:
	// filename_or_dir_1filename_or_dir_2filename_or_dir_3filename_or_dir_4...
	// 
	CkTask *NlstXmlAsync(const char *remoteDirPattern);


	// Issues a no-op command to the FTP server.
	bool Noop(void);

	// Issues a no-op command to the FTP server.
	CkTask *NoopAsync(void);


	// Uploads a local file to the current directory on the FTP server.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool PutFile(const char *localFilePath, const char *remoteFilePath);

	// Uploads a local file to the current directory on the FTP server.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *PutFileAsync(const char *localFilePath, const char *remoteFilePath);


	// Uploads the contents of a BinData to a remote file.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool PutFileBd(CkBinData &binData, const char *remoteFilePath);

	// Uploads the contents of a BinData to a remote file.
	// 
	// If the remoteFilePath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *PutFileBdAsync(CkBinData &binData, const char *remoteFilePath);


	// Creates a file on the remote server containing the data passed in a byte array.
	bool PutFileFromBinaryData(const char *remoteFilename, CkByteData &content);

	// Creates a file on the remote server containing the data passed in a byte array.
	CkTask *PutFileFromBinaryDataAsync(const char *remoteFilename, CkByteData &content);


	// Creates a file on the remote server containing the data passed in a string.
	bool PutFileFromTextData(const char *remoteFilename, const char *textData, const char *charset);

	// Creates a file on the remote server containing the data passed in a string.
	CkTask *PutFileFromTextDataAsync(const char *remoteFilename, const char *textData, const char *charset);


	// Uploads the contents of a StringBuilder to a remote file.
	// 
	// If the charset contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool PutFileSb(CkStringBuilder &sb, const char *charset, bool includeBom, const char *remoteFilePath);

	// Uploads the contents of a StringBuilder to a remote file.
	// 
	// If the charset contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *PutFileSbAsync(CkStringBuilder &sb, const char *charset, bool includeBom, const char *remoteFilePath);


	// Executes an "FTP plan" (created by the CreatePlan method) and logs each
	// successful operation to a plan log file. If a large-scale upload is interrupted,
	// the PutPlan can be resumed, skipping over the operations already listed in the
	// plan log file. When resuming an interrupted PutPlan method, use the same log
	// file. All completed operations found in the already-existing log will
	// automatically be skipped.
	bool PutPlan(const char *plan, const char *alreadyDoneFilename);

	// Executes an "FTP plan" (created by the CreatePlan method) and logs each
	// successful operation to a plan log file. If a large-scale upload is interrupted,
	// the PutPlan can be resumed, skipping over the operations already listed in the
	// plan log file. When resuming an interrupted PutPlan method, use the same log
	// file. All completed operations found in the already-existing log will
	// automatically be skipped.
	CkTask *PutPlanAsync(const char *plan, const char *alreadyDoneFilename);


	// Uploads an entire directory tree from the local filesystem to the remote FTP
	// server, recreating the directory tree on the server. The PutTree method copies a
	// directory tree to the current remote directory on the FTP server.
	bool PutTree(const char *localDir);

	// Uploads an entire directory tree from the local filesystem to the remote FTP
	// server, recreating the directory tree on the server. The PutTree method copies a
	// directory tree to the current remote directory on the FTP server.
	CkTask *PutTreeAsync(const char *localDir);


	// Sends an arbitrary (raw) command to the FTP server.
	bool Quote(const char *cmd);

	// Sends an arbitrary (raw) command to the FTP server.
	CkTask *QuoteAsync(const char *cmd);


	// Removes a directory from the FTP server.
	// 
	// If the remoteDirPath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	bool RemoveRemoteDir(const char *remoteDirPath);

	// Removes a directory from the FTP server.
	// 
	// If the remoteDirPath contains non-English characters, it may be necessary to set the
	// DirListingCharset property equal to "utf-8". Please refer to the documentation
	// for the DirListingCharset property.
	// 
	CkTask *RemoveRemoteDirAsync(const char *remoteDirPath);


	// Renames a file or directory on the FTP server. To move a file from one directory
	// to another on a remote FTP server, call this method and include the source and
	// destination directory filepath.
	// 
	// If the existingRemoteFilePath or newRemoteFilePath contains non-English characters, it may be necessary to set
	// the DirListingCharset property equal to "utf-8". Please refer to the
	// documentation for the DirListingCharset property.
	// 
	bool RenameRemoteFile(const char *existingRemoteFilePath, const char *newRemoteFilePath);

	// Renames a file or directory on the FTP server. To move a file from one directory
	// to another on a remote FTP server, call this method and include the source and
	// destination directory filepath.
	// 
	// If the existingRemoteFilePath or newRemoteFilePath contains non-English characters, it may be necessary to set
	// the DirListingCharset property equal to "utf-8". Please refer to the
	// documentation for the DirListingCharset property.
	// 
	CkTask *RenameRemoteFileAsync(const char *existingRemoteFilePath, const char *newRemoteFilePath);


	// Sends an raw command to the FTP server and returns the raw response.
	bool SendCommand(const char *cmd, CkString &outReply);

	// Sends an raw command to the FTP server and returns the raw response.
	const char *sendCommand(const char *cmd);
	// Sends an raw command to the FTP server and returns the raw response.
	CkTask *SendCommandAsync(const char *cmd);


	// Chilkat FTP2 supports MODE Z, which is a transfer mode implemented by some FTP
	// servers. It allows for files to be uploaded and downloaded using compressed
	// streams (using the zlib deflate algorithm).
	// 
	// Call this method after connecting to enable Mode Z. Once enabled, all transfers
	// (uploads, downloads, and directory listings) are compressed.
	// 
	bool SetModeZ(void);

	// Chilkat FTP2 supports MODE Z, which is a transfer mode implemented by some FTP
	// servers. It allows for files to be uploaded and downloaded using compressed
	// streams (using the zlib deflate algorithm).
	// 
	// Call this method after connecting to enable Mode Z. Once enabled, all transfers
	// (uploads, downloads, and directory listings) are compressed.
	// 
	CkTask *SetModeZAsync(void);


	// Used in conjunction with the DownloadTree method. Call this method prior to
	// calling DownloadTree to set the oldest date for a file to be downloaded. When
	// DownloadTree is called, any file older than this date will not be downloaded.
	void SetOldestDate(SYSTEMTIME &oldestDateTime);


	// Used in conjunction with the DownloadTree method. Call this method prior to
	// calling DownloadTree to set the oldest date for a file to be downloaded. When
	// DownloadTree is called, any file older than this date will not be downloaded.
	// 
	// The oldestDateTimeStr should be a date/time string in RFC822 format, such as "Tue, 25 Sep
	// 2012 12:25:32 -0500".
	// 
	void SetOldestDateStr(const char *oldestDateTimeStr);


	// This is a general purpose method to set miscellaneous options that might arise
	// due to buggy or quirky FTP servers. The option is a string describing the option.
	// The current list of possible options are:
	//     "Microsoft-TLS-1.2-Workaround" -- This is to force the data connection to
	//     use TLS 1.0 instead of the default. It works around the Microsoft FTP server bug
	//     found here: https://support.microsoft.com/en-us/kb/2888853
	// 
	// To turn off an option, prepend the string "No-". For example
	// "No-Microsoft-TLS-1.2-Workaround". All options are turned off by default.
	// 
	bool SetOption(const char *option);


	// Sets the password in a more secure way than setting the Password property.
	// Calling this method is the equivalent of setting the Password property.
	// 
	// Note: Starting in v9.5.0.76, this method has been copied to SetSecurePassword.
	// Applications should call SetSecurePassword instead because this method is now
	// deprecated.
	// 
	bool SetPassword(CkSecureString &password);


	// Sets the last-modified date/time of a file on the FTP server. Important: Not all
	// FTP servers support this functionality. Please see the information at the
	// Chilkat blog below:
	bool SetRemoteFileDateTime(SYSTEMTIME &dt, const char *remoteFilename);


	// Sets the last-modified date/time of a file on the FTP server. The dateTimeStr should be
	// a date/time string in RFC822 format, such as "Tue, 25 Sep 2012 12:25:32 -0500".
	// 
	// Important: Not all FTP servers support this functionality. Please see the
	// information at the Chilkat blog below:
	// 
	bool SetRemoteFileDateTimeStr(const char *dateTimeStr, const char *remoteFilename);

	// Sets the last-modified date/time of a file on the FTP server. The dateTimeStr should be
	// a date/time string in RFC822 format, such as "Tue, 25 Sep 2012 12:25:32 -0500".
	// 
	// Important: Not all FTP servers support this functionality. Please see the
	// information at the Chilkat blog below:
	// 
	CkTask *SetRemoteFileDateTimeStrAsync(const char *dateTimeStr, const char *remoteFilename);


	// Sets the last-modified date/time of a file on the FTP server. Important: Not all
	// FTP servers support this functionality. Please see the information at the
	// Chilkat blog below:
	bool SetRemoteFileDt(CkDateTime &dt, const char *remoteFilename);

	// Sets the last-modified date/time of a file on the FTP server. Important: Not all
	// FTP servers support this functionality. Please see the information at the
	// Chilkat blog below:
	CkTask *SetRemoteFileDtAsync(CkDateTime &dt, const char *remoteFilename);


	// Sets the password in a more secure way than setting the Password property.
	// Calling this method is the equivalent of setting the Password property.
	bool SetSecurePassword(CkSecureString &password);


	// Enforces a requirement on the server's certificate. The reqName can be one of the
	// following:
	//     SubjectDN
	//     SubjectCN
	//     IssuerDN
	//     IssuerCN
	//     SAN (added in v9.5.0.84)
	// 
	// The reqName specifies the part of the certificate, and the reqValue is the value that
	// it must match exactly or with a wildcard (*). If the server's certificate does
	// not match, the SSL / TLS connection is aborted.
	// 
	void SetSslCertRequirement(const char *reqName, const char *reqValue);


	// Allows for a client-side certificate to be used for the SSL / TLS connection.
	bool SetSslClientCert(CkCert &cert);


	// Allows for a client-side certificate to be used for the SSL / TLS connection. If
	// the PEM requires no password, pass an empty string in pemPassword. If the PEM is in a
	// file, pass the path to the file in pemDataOrFilename. If the PEM is already loaded into a
	// string variable, then pass the string containing the contents of the PEM in
	// pemDataOrFilename.
	bool SetSslClientCertPem(const char *pemDataOrFilename, const char *pemPassword);


	// Allows for a client-side certificate to be used for the SSL / TLS connection.
	bool SetSslClientCertPfx(const char *pfxFilename, const char *pfxPassword);


	// Set the FTP transfer mode to us-ascii.
	bool SetTypeAscii(void);

	// Set the FTP transfer mode to us-ascii.
	CkTask *SetTypeAsciiAsync(void);


	// Set the FTP transfer mode to binary.
	bool SetTypeBinary(void);

	// Set the FTP transfer mode to binary.
	CkTask *SetTypeBinaryAsync(void);


	// Sends an arbitrary "site" command to the FTP server. The params argument should
	// contain the parameters to the site command as they would appear on a command
	// line. For example: "recfm=fb lrecl=600".
	bool Site(const char *siteCommand);

	// Sends an arbitrary "site" command to the FTP server. The params argument should
	// contain the parameters to the site command as they would appear on a command
	// line. For example: "recfm=fb lrecl=600".
	CkTask *SiteAsync(const char *siteCommand);


	// Causes the calling process to sleep for a number of milliseconds.
	void SleepMs(int millisec);


	// Sends a STAT command to the FTP server and returns the server's reply.
	bool Stat(CkString &outStr);

	// Sends a STAT command to the FTP server and returns the server's reply.
	const char *ck_stat(void);
	// Sends a STAT command to the FTP server and returns the server's reply.
	CkTask *StatAsync(void);


	// Delete remote files that do not exist locally. The remote directory tree rooted
	// at the current remote directory is traversed and remote files that have no
	// corresponding local file are deleted.
	// 
	// Note: In v9.5.0.51 and higher, the list of deleted files is available in the
	// SyncedFiles property.
	// 
	bool SyncDeleteRemote(const char *localRoot);

	// Delete remote files that do not exist locally. The remote directory tree rooted
	// at the current remote directory is traversed and remote files that have no
	// corresponding local file are deleted.
	// 
	// Note: In v9.5.0.51 and higher, the list of deleted files is available in the
	// SyncedFiles property.
	// 
	CkTask *SyncDeleteRemoteAsync(const char *localRoot);


	// The same as SyncLocalTree, except the sub-directories are not traversed. The
	// files in the current remote directory are synchronized (downloaded) with the
	// files in localRoot. For possible mode settings, see SyncLocalTree.
	// 
	// Note: In v9.5.0.51 and higher, the list of downloaded files is available in the
	// SyncedFiles property.
	// 
	bool SyncLocalDir(const char *localRoot, int mode);

	// The same as SyncLocalTree, except the sub-directories are not traversed. The
	// files in the current remote directory are synchronized (downloaded) with the
	// files in localRoot. For possible mode settings, see SyncLocalTree.
	// 
	// Note: In v9.5.0.51 and higher, the list of downloaded files is available in the
	// SyncedFiles property.
	// 
	CkTask *SyncLocalDirAsync(const char *localRoot, int mode);


	// Downloads files from the FTP server to a local directory tree. Synchronization
	// modes include:
	// 
	//     mode=0: Download all files
	//     mode=1: Download all files that do not exist on the local filesystem.
	//     mode=2: Download newer or non-existant files.
	//     mode=3: Download only newer files. If a file does not already exist on the
	//     local filesystem, it is not downloaded from the server.
	//     mode=5: Download only missing files or files with size differences.
	//     mode=6: Same as mode 5, but also download newer files.
	//     mode=99: Do not download files, but instead delete remote files that do not
	//     exist locally.
	//     * There is no mode #4. It is a mode used internally by the DirTreeXml
	//     method.
	//     
	// 
	// Note: In v9.5.0.51 and higher, the list of downloaded (or deleted) files is
	// available in the SyncedFiles property.
	// 
	bool SyncLocalTree(const char *localRoot, int mode);

	// Downloads files from the FTP server to a local directory tree. Synchronization
	// modes include:
	// 
	//     mode=0: Download all files
	//     mode=1: Download all files that do not exist on the local filesystem.
	//     mode=2: Download newer or non-existant files.
	//     mode=3: Download only newer files. If a file does not already exist on the
	//     local filesystem, it is not downloaded from the server.
	//     mode=5: Download only missing files or files with size differences.
	//     mode=6: Same as mode 5, but also download newer files.
	//     mode=99: Do not download files, but instead delete remote files that do not
	//     exist locally.
	//     * There is no mode #4. It is a mode used internally by the DirTreeXml
	//     method.
	//     
	// 
	// Note: In v9.5.0.51 and higher, the list of downloaded (or deleted) files is
	// available in the SyncedFiles property.
	// 
	CkTask *SyncLocalTreeAsync(const char *localRoot, int mode);


	// Uploads a directory tree from the local filesystem to the FTP server.
	// Synchronization modes include:
	// 
	//     mode=0: Upload all files
	//     mode=1: Upload all files that do not exist on the FTP server.
	//     mode=2: Upload newer or non-existant files.
	//     mode=3: Upload only newer files. If a file does not already exist on the FTP
	//     server, it is not uploaded.
	//     mode=4: transfer missing files or files with size differences.
	//     mode=5: same as mode 4, but also newer files.
	// 
	// Note: In v9.5.0.51 and higher, the list of uploaded files is available in the
	// SyncedFiles property.
	// 
	bool SyncRemoteTree(const char *localRoot, int mode);

	// Uploads a directory tree from the local filesystem to the FTP server.
	// Synchronization modes include:
	// 
	//     mode=0: Upload all files
	//     mode=1: Upload all files that do not exist on the FTP server.
	//     mode=2: Upload newer or non-existant files.
	//     mode=3: Upload only newer files. If a file does not already exist on the FTP
	//     server, it is not uploaded.
	//     mode=4: transfer missing files or files with size differences.
	//     mode=5: same as mode 4, but also newer files.
	// 
	// Note: In v9.5.0.51 and higher, the list of uploaded files is available in the
	// SyncedFiles property.
	// 
	CkTask *SyncRemoteTreeAsync(const char *localRoot, int mode);


	// Same as SyncRemoteTree, except two extra arguments are added to allow for more
	// flexibility. If bDescend is false, then the directory tree is not descended and
	// only the files in localDirPath are synchronized. If bPreviewOnly is true then no files are
	// transferred and instead the files that would've been transferred (had bPreviewOnly been
	// set to false) are listed in the SyncPreview property.
	// 
	// Note: If bPreviewOnly is set to true, the remote directories (if they do not exist)
	// are created. It is only the files that are not uploaded.
	// 
	// Note: In v9.5.0.51 and higher, the list of uploaded files is available in the
	// SyncedFiles property.
	// 
	bool SyncRemoteTree2(const char *localDirPath, int mode, bool bDescend, bool bPreviewOnly);

	// Same as SyncRemoteTree, except two extra arguments are added to allow for more
	// flexibility. If bDescend is false, then the directory tree is not descended and
	// only the files in localDirPath are synchronized. If bPreviewOnly is true then no files are
	// transferred and instead the files that would've been transferred (had bPreviewOnly been
	// set to false) are listed in the SyncPreview property.
	// 
	// Note: If bPreviewOnly is set to true, the remote directories (if they do not exist)
	// are created. It is only the files that are not uploaded.
	// 
	// Note: In v9.5.0.51 and higher, the list of uploaded files is available in the
	// SyncedFiles property.
	// 
	CkTask *SyncRemoteTree2Async(const char *localDirPath, int mode, bool bDescend, bool bPreviewOnly);


	// Sends a SYST command to the FTP server to find out the type of operating system
	// at the server. The method returns the FTP server's response string. Refer to RFC
	// 959 for details.
	bool Syst(CkString &outStr);

	// Sends a SYST command to the FTP server to find out the type of operating system
	// at the server. The method returns the FTP server's response string. Refer to RFC
	// 959 for details.
	const char *syst(void);
	// Sends a SYST command to the FTP server to find out the type of operating system
	// at the server. The method returns the FTP server's response string. Refer to RFC
	// 959 for details.
	CkTask *SystAsync(void);


	// Unlocks the component. This must be called once prior to calling any other
	// method. A purchased unlock code for FTP2 should contain the substring "FTP", or
	// can be a Bundle unlock code.
	bool UnlockComponent(const char *unlockCode);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
