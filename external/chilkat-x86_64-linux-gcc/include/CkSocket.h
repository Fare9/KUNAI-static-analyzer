// CkSocket.h: interface for the CkSocket class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkSocket_H
#define _CkSocket_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkTask;
class CkCert;
class CkJsonObject;
class CkBinData;
class CkByteData;
class CkStringBuilder;
class CkSshKey;
class CkSsh;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkSocket
class CK_VISIBLE_PUBLIC CkSocket  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkSocket(const CkSocket &);
	CkSocket &operator=(const CkSocket &);

    public:
	CkSocket(void);
	virtual ~CkSocket(void);

	static CkSocket *createNew(void);
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

	// If a AcceptNextConnection method fails, this property can be checked to
	// determine the reason for failure.
	// 
	// Note: If accepting a TLS connection, then this property can also have any of the
	// values listed for the ReceiveFailReason and SendFailReason properties (because
	// the TLS handshake involves sending/receiving on the initial TCP socket).
	// 
	// Possible values are:
	// 
	// 0 = Success
	// 1 = An async operation is  in progress.
	// 3 = An unspecified internal failure, perhaps out-of-memory, caused the failure.
	// 5 = Timeout.  No connections were accepted in the amount of time alotted.
	// 6 = The receive was aborted by the application in an event callback.
	// 9 = An unspecified fatal socket error occurred (less common).
	// 20 = Must first bind and listen on a port.
	// 99 = The component is not unlocked.
	// 
	// Errors Relating to the SSL/TLS Handshake:
	// 100 = TLS internal error.
	// 102 = Unexpected handshake message.
	// 109 = Failed to read handshake messages.
	// 114 = Failed to send change cipher spec handshake message.
	// 115 = Failed to send finished handshake message.
	// 116 = Client's Finished message is invalid.
	// 117 = Unable to agree on TLS protocol version.
	// 118 = Unable to agree on a cipher spec.
	// 119 = Failed to read the client's hello message.
	// 120 = Failed to send handshake messages.
	// 121 = Failed to process client cert message.
	// 122 = Failed to process client cert URL message.
	// 123 = Failed to process client key exchange message.
	// 124 = Failed to process certificate verify message.
	// 125 = Received and rejected an SSL 2.0 connection attempt.
	int get_AcceptFailReason(void);

	// For TLS connections. Can be set to the name of an application layer protocol.
	// This causes the ALPN extension to be added to the TLS ClientHello with the given
	// ALPN protocol name.
	void get_AlpnProtocol(CkString &str);
	// For TLS connections. Can be set to the name of an application layer protocol.
	// This causes the ALPN extension to be added to the TLS ClientHello with the given
	// ALPN protocol name.
	const char *alpnProtocol(void);
	// For TLS connections. Can be set to the name of an application layer protocol.
	// This causes the ALPN extension to be added to the TLS ClientHello with the given
	// ALPN protocol name.
	void put_AlpnProtocol(const char *newVal);

	// If non-zero, limits (throttles) the receiving bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	int get_BandwidthThrottleDown(void);
	// If non-zero, limits (throttles) the receiving bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	void put_BandwidthThrottleDown(int newVal);

	// If non-zero, limits (throttles) the sending bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	int get_BandwidthThrottleUp(void);
	// If non-zero, limits (throttles) the sending bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	void put_BandwidthThrottleUp(int newVal);

	// Applies to the SendCount and ReceiveCount methods. If BigEndian is set to true
	// (the default) then the 4-byte count is in big endian format. Otherwise it is
	// little endian.
	bool get_BigEndian(void);
	// Applies to the SendCount and ReceiveCount methods. If BigEndian is set to true
	// (the default) then the 4-byte count is in big endian format. Otherwise it is
	// little endian.
	void put_BigEndian(bool newVal);

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

	// Normally left at the default value of 0, in which case a unique port is assigned
	// with a value between 1024 and 5000. This property would only be changed if it is
	// specifically required. For example, one customer's requirements are as follows:
	// 
	//     "I have to connect to a Siemens PLC IP server on a technical network. This
	//     machine expects that I connect to its server from a specific IP address using a
	//     specific port otherwise the build in security disconnect the IP connection."
	// 
	int get_ClientPort(void);
	// Normally left at the default value of 0, in which case a unique port is assigned
	// with a value between 1024 and 5000. This property would only be changed if it is
	// specifically required. For example, one customer's requirements are as follows:
	// 
	//     "I have to connect to a Siemens PLC IP server on a technical network. This
	//     machine expects that I connect to its server from a specific IP address using a
	//     specific port otherwise the build in security disconnect the IP connection."
	// 
	void put_ClientPort(int newVal);

	// If the Connect method fails, this property can be checked to determine the
	// reason for failure.
	// 
	// Possible values are:
	// 0 = success
	// 
	// Normal (non-SSL) sockets:
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
	// 108 = App-defined server certificate requirements failure.
	// 109 = Failed to read handshake messages.
	// 110 = Failed to send client certificate handshake message.
	// 111 = Failed to send client key exchange handshake message.
	// 112 = Client certificate's private key not accessible.
	// 113 = Failed to send client cert verify handshake message.
	// 114 = Failed to send change cipher spec handshake message.
	// 115 = Failed to send finished handshake message.
	// 116 = Server's Finished message is invalid.
	// 
	int get_ConnectFailReason(void);

	// Contains the number of seconds since the last call to StartTiming, otherwise
	// contains 0. (The StartTiming method and ElapsedSeconds property is provided for
	// convenience.)
	int get_ElapsedSeconds(void);

	// The number of milliseconds between periodic heartbeat callbacks for blocking
	// socket operations (connect, accept, dns query, send, receive). Set this to 0 to
	// disable heartbeat events. The default value is 1000 (i.e. 1 heartbeat callback
	// per second).
	int get_HeartbeatMs(void);
	// The number of milliseconds between periodic heartbeat callbacks for blocking
	// socket operations (connect, accept, dns query, send, receive). Set this to 0 to
	// disable heartbeat events. The default value is 1000 (i.e. 1 heartbeat callback
	// per second).
	void put_HeartbeatMs(int newVal);

	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	void get_HttpProxyAuthMethod(CkString &str);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	const char *httpProxyAuthMethod(void);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	void put_HttpProxyAuthMethod(const char *newVal);

	// The NTLM authentication domain (optional) if NTLM authentication is used.
	void get_HttpProxyDomain(CkString &str);
	// The NTLM authentication domain (optional) if NTLM authentication is used.
	const char *httpProxyDomain(void);
	// The NTLM authentication domain (optional) if NTLM authentication is used.
	void put_HttpProxyDomain(const char *newVal);

	// If this connection is effectively used to send HTTP requests, then set this
	// property to true when using an HTTP proxy. The default value of this property
	// is false.
	// 
	// This is because an HTTP proxy used for other protocols (IMAP, SMTP, SSH, FTP,
	// etc.) can require some internal differences in behavior (i.e. how we do things).
	// 
	// For example, the Chilkat REST object can use this socket object's connection via
	// the UseConnection method. This is a case where we know the proxied connection is
	// for the HTTP protocol. Therefore we should set this property to true. (See the
	// example below.)
	// 
	bool get_HttpProxyForHttp(void);
	// If this connection is effectively used to send HTTP requests, then set this
	// property to true when using an HTTP proxy. The default value of this property
	// is false.
	// 
	// This is because an HTTP proxy used for other protocols (IMAP, SMTP, SSH, FTP,
	// etc.) can require some internal differences in behavior (i.e. how we do things).
	// 
	// For example, the Chilkat REST object can use this socket object's connection via
	// the UseConnection method. This is a case where we know the proxied connection is
	// for the HTTP protocol. Therefore we should set this property to true. (See the
	// example below.)
	// 
	void put_HttpProxyForHttp(bool newVal);

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

	// Returns true if the socket is connected. Otherwise returns false.
	// 
	// Note: In general, this property indicates the last known state of the socket.
	// For example, if the socket is connected, and your application does not read or
	// write the socket, then IsConnected will remain true. This property is updated
	// when your application tries to read or write and discovers that the socket is no
	// longer connected. It is also updated if your application explicitly closes the
	// socket.
	// 
	bool get_IsConnected(void);

	// Controls whether the SO_KEEPALIVE socket option is used for the underlying
	// TCP/IP socket. The default value is true.
	bool get_KeepAlive(void);
	// Controls whether the SO_KEEPALIVE socket option is used for the underlying
	// TCP/IP socket. The default value is true.
	void put_KeepAlive(bool newVal);

	// Controls whether socket (or SSL) communications are logged to the SessionLog
	// string property. To turn on session logging, set this property = true,
	// otherwise set to false (which is the default value).
	bool get_KeepSessionLog(void);
	// Controls whether socket (or SSL) communications are logged to the SessionLog
	// string property. To turn on session logging, set this property = true,
	// otherwise set to false (which is the default value).
	void put_KeepSessionLog(bool newVal);

	// true if the last method called on this object failed. This provides an easier
	// (less confusing) way of determining whether a method such as ReceiveBytes
	// succeeded or failed.
	bool get_LastMethodFailed(void);

	// If set to true, then a socket that listens for incoming connections (via the
	// BindAndList and AcceptNextConnection method calls) will use IPv6 and not IPv4.
	// The default value is false for IPv4.
	bool get_ListenIpv6(void);
	// If set to true, then a socket that listens for incoming connections (via the
	// BindAndList and AcceptNextConnection method calls) will use IPv6 and not IPv4.
	// The default value is false for IPv4.
	void put_ListenIpv6(bool newVal);

	// The BindAndListen method will find a random unused port to listen on if you bind
	// to port 0. This chosen listen port is available via this property.
	int get_ListenPort(void);

	// The local IP address for a bound or connected socket.
	void get_LocalIpAddress(CkString &str);
	// The local IP address for a bound or connected socket.
	const char *localIpAddress(void);

	// The local port for a bound or connected socket.
	int get_LocalPort(void);

	// The maximum number of milliseconds to wait on a socket read operation while no
	// additional data is forthcoming. To wait indefinitely, set this property to 0.
	// The default value is 0.
	int get_MaxReadIdleMs(void);
	// The maximum number of milliseconds to wait on a socket read operation while no
	// additional data is forthcoming. To wait indefinitely, set this property to 0.
	// The default value is 0.
	void put_MaxReadIdleMs(int newVal);

	// The maximum number of milliseconds to wait for the socket to become writeable on
	// a socket write operation. To wait indefinitely, set this property to 0. The
	// default value is 0.
	int get_MaxSendIdleMs(void);
	// The maximum number of milliseconds to wait for the socket to become writeable on
	// a socket write operation. To wait indefinitely, set this property to 0. The
	// default value is 0.
	void put_MaxSendIdleMs(int newVal);

	// The local IP address of the local computer. For multi-homed computers (i.e.
	// computers with multiple IP adapters) this property returns the default IP
	// address.
	// 
	// Note: This will be the internal IP address, not an external IP address. (For
	// example, if your computer is on a LAN, it is likely to be an IP address
	// beginning with "192.168.".
	// 
	// Important: Use LocalIpAddress and LocalIpPort to get the local IP/port for a
	// bound or connected socket.
	// 
	void get_MyIpAddress(CkString &str);
	// The local IP address of the local computer. For multi-homed computers (i.e.
	// computers with multiple IP adapters) this property returns the default IP
	// address.
	// 
	// Note: This will be the internal IP address, not an external IP address. (For
	// example, if your computer is on a LAN, it is likely to be an IP address
	// beginning with "192.168.".
	// 
	// Important: Use LocalIpAddress and LocalIpPort to get the local IP/port for a
	// bound or connected socket.
	// 
	const char *myIpAddress(void);

	// If the socket is the server-side of an SSL/TLS connection, the property
	// represents the number of client-side certificates received during the SSL/TLS
	// handshake (i.e. connection process). Each client-side cert may be retrieved by
	// calling the GetReceivedClientCert method and passing an integer index value from
	// 0 to N-1, where N is the number of client certs received.
	// 
	// Note: A client only sends a certificate if 2-way SSL/TLS is required. In other
	// words, if the server demands a certificate from the client.
	// 
	// Important: This property should be examined on the socket object that is
	// returned by AcceptNextConnection.
	// 
	int get_NumReceivedClientCerts(void);

	// If this socket is a "socket set", then NumSocketsInSet returns the number of
	// sockets contained in the set. A socket object can become a "socket set" by
	// calling the TakeSocket method on one or more connected sockets. This makes it
	// possible to select for reading on the set (i.e. wait for data to arrive from any
	// one of multiple sockets). See the following methods and properties for more
	// information: TakeSocket, SelectorIndex, SelectorReadIndex, SelectorWriteIndex,
	// SelectForReading, SelectForWriting.
	int get_NumSocketsInSet(void);

	// If connected as an SSL/TLS client to an SSL/TLS server where the server requires
	// a client-side certificate for authentication, then this property contains the
	// number of acceptable certificate authorities sent by the server during
	// connection establishment handshake. The GetSslAcceptableClientCaDn method may be
	// called to get the Distinguished Name (DN) of each acceptable CA.
	int get_NumSslAcceptableClientCAs(void);

	// Each socket object is assigned a unique object ID. This ID is passed in event
	// callbacks to allow your application to associate the event with the socket
	// object.
	int get_ObjectId(void);

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

	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	bool get_PreferIpv6(void);
	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	void put_PreferIpv6(bool newVal);

	// Returns the cumulative receive rate in bytes per second. The measurement
	// includes the overhead bytes for protocols such as TLS or SSH tunneling. For
	// example, if 1000 application bytes are received, the actual number of raw bytes
	// received on a TLS connection is greater. This property measures the actual
	// number of raw bytes received in a given time period. The ResetPerf method can be
	// called to reset this property value and to begin the performance measurement
	// afresh.
	int get_RcvBytesPerSec(void);

	// Any method that receives data will increase the value of this property by the
	// number of bytes received. The application may reset this property to 0 at any
	// point. It is provided as a way to keep count of the total number of bytes
	// received on a socket connection, regardless of which method calls are used to
	// receive the data.
	// 
	// Note: The ReceivedCount may be larger than the number of bytes returned by some
	// methods. For methods such as ReceiveUntilMatch, the excess received on the
	// socket (beyond the match), is buffered by Chilkat for subsequent method calls.
	// The ReceivedCount is updated based on the actual number of bytes received on the
	// underlying socket in real-time. (The ReceivedCount does not include the overhead
	// bytes associated with the TLS and/or SSH protocols.
	// 
	int get_ReceivedCount(void);
	// Any method that receives data will increase the value of this property by the
	// number of bytes received. The application may reset this property to 0 at any
	// point. It is provided as a way to keep count of the total number of bytes
	// received on a socket connection, regardless of which method calls are used to
	// receive the data.
	// 
	// Note: The ReceivedCount may be larger than the number of bytes returned by some
	// methods. For methods such as ReceiveUntilMatch, the excess received on the
	// socket (beyond the match), is buffered by Chilkat for subsequent method calls.
	// The ReceivedCount is updated based on the actual number of bytes received on the
	// underlying socket in real-time. (The ReceivedCount does not include the overhead
	// bytes associated with the TLS and/or SSH protocols.
	// 
	void put_ReceivedCount(int newVal);

	// Contains the last integer received via a call to ReceiveByte, ReceiveInt16, or
	// ReceiveInt32.
	int get_ReceivedInt(void);
	// Contains the last integer received via a call to ReceiveByte, ReceiveInt16, or
	// ReceiveInt32.
	void put_ReceivedInt(int newVal);

	// If a Receive method fails, this property can be checked to determine the reason
	// for failure.
	// 
	// Possible values are:
	// 0 = Success
	// 1 = An async receive operation is already in progress.
	// 2 = The socket is not connected, such as if it was never connected, or if the connection was previously lost.
	// 3 = An unspecified internal failure, perhaps out-of-memory, caused the failure.
	// 4 = Invalid parameters were passed to the receive method call.
	// 5 = Timeout.  Data stopped arriving for more than the amount of time specified by the MaxReadIdleMs property.
	// 6 = The receive was aborted by the application in an event callback.
	// 7 = The connection was lost -- the remote peer reset the connection. (The connection was forcibly closed by the peer.)
	// 8 = An established connection was aborted by the software in your host machine. (See https://www.chilkatsoft.com/p/p_299.asp )
	// 9 = An unspecified fatal socket error occurred (less common).
	// 10 = The connection was closed by the peer.
	// 
	int get_ReceiveFailReason(void);

	// The number of bytes to receive at a time (internally). This setting has an
	// effect on methods such as ReadBytes and ReadString where the number of bytes to
	// read is not explicitly specified. The default value is 4096.
	int get_ReceivePacketSize(void);
	// The number of bytes to receive at a time (internally). This setting has an
	// effect on methods such as ReadBytes and ReadString where the number of bytes to
	// read is not explicitly specified. The default value is 4096.
	void put_ReceivePacketSize(int newVal);

	// When a socket is connected, the remote IP address of the connected peer is
	// available in this property.
	void get_RemoteIpAddress(CkString &str);
	// When a socket is connected, the remote IP address of the connected peer is
	// available in this property.
	const char *remoteIpAddress(void);

	// When a socket is connected, the remote port of the connected peer is available
	// in this property.
	int get_RemotePort(void);

	// If true, then the SSL/TLS client will verify the server's SSL certificate. The
	// certificate is expired, or if the cert's signature is invalid, the connection is
	// not allowed. The default value of this property is false.
	bool get_RequireSslCertVerify(void);
	// If true, then the SSL/TLS client will verify the server's SSL certificate. The
	// certificate is expired, or if the cert's signature is invalid, the connection is
	// not allowed. The default value of this property is false.
	void put_RequireSslCertVerify(bool newVal);

	// If this socket contains a collection of connected sockets (i.e. it is a "socket
	// set") then method calls and property gets/sets are routed to the contained
	// socket indicated by this property. Indexing begins at 0. See the TakeSocket
	// method and SelectForReading method for more information.
	int get_SelectorIndex(void);
	// If this socket contains a collection of connected sockets (i.e. it is a "socket
	// set") then method calls and property gets/sets are routed to the contained
	// socket indicated by this property. Indexing begins at 0. See the TakeSocket
	// method and SelectForReading method for more information.
	void put_SelectorIndex(int newVal);

	// When SelectForReading returns a number greater than 0 indicating that 1 or more
	// sockets are ready for reading, this property is used to select the socket in the
	// "ready set" for reading. See the example below:
	int get_SelectorReadIndex(void);
	// When SelectForReading returns a number greater than 0 indicating that 1 or more
	// sockets are ready for reading, this property is used to select the socket in the
	// "ready set" for reading. See the example below:
	void put_SelectorReadIndex(int newVal);

	// When SelectForWriting returns a number greater than 0 indicating that one or
	// more sockets are ready for writing, this property is used to select the socket
	// in the "ready set" for writing.
	int get_SelectorWriteIndex(void);
	// When SelectForWriting returns a number greater than 0 indicating that one or
	// more sockets are ready for writing, this property is used to select the socket
	// in the "ready set" for writing.
	void put_SelectorWriteIndex(int newVal);

	// Returns the cumulative send rate in bytes per second. The measurement includes
	// the overhead bytes for protocols such as TLS or SSH tunneling. For example, if
	// 1000 application bytes are sent, the actual number of raw bytes sent on a TLS
	// connection is greater. This property measures the actual number of raw bytes
	// sent in a given time period. The ResetPerf method can be called to reset this
	// property value and to begin the performance measurement afresh.
	int get_SendBytesPerSec(void);

	// If a Send method fails, this property can be checked to determine the reason for
	// failure.
	// 
	// Possible values are:
	// 0 = Success
	// 1 = An async receive operation is already in progress.
	// 2 = The socket is not connected, such as if it was never connected, or if the connection was previously lost.
	// 3 = An unspecified internal failure, perhaps out-of-memory, caused the failure.
	// 4 = Invalid parameters were passed to the receive method call.
	// 5 = Timeout.  Data stopped arriving for more than the amount of time specified by the MaxReadIdleMs property.
	// 6 = The receive was aborted by the application in an event callback.
	// 7 = The connection was lost -- the remote peer reset the connection. (The connection was forcibly closed by the peer.)
	// 8 = An established connection was aborted by the software in your host machine. (See https://www.chilkatsoft.com/p/p_299.asp )
	// 9 = An unspecified fatal socket error occurred (less common).
	// 10 = The connection was closed by the peer.
	// 11 = Decoding error (possible in SendString when coverting to the StringCharset, or in SendBytesENC).
	// 
	int get_SendFailReason(void);

	// The number of bytes to send at a time (internally). This can also be though of
	// as the "chunk size". If a large amount of data is to be sent, the data is sent
	// in chunks equal to this size in bytes. The default value is 65535. (Note: This
	// only applies to non-SSL/TLS connections. SSL and TLS have their own pre-defined
	// packet sizes.)
	int get_SendPacketSize(void);
	// The number of bytes to send at a time (internally). This can also be though of
	// as the "chunk size". If a large amount of data is to be sent, the data is sent
	// in chunks equal to this size in bytes. The default value is 65535. (Note: This
	// only applies to non-SSL/TLS connections. SSL and TLS have their own pre-defined
	// packet sizes.)
	void put_SendPacketSize(int newVal);

	// Contains a log of the bytes sent and received on this socket. The KeepSessionLog
	// property must be set to true for logging to occur.
	void get_SessionLog(CkString &str);
	// Contains a log of the bytes sent and received on this socket. The KeepSessionLog
	// property must be set to true for logging to occur.
	const char *sessionLog(void);

	// Controls how the data is encoded in the SessionLog. Possible values are "esc"
	// and "hex". The default value is "esc".
	// 
	// When set to "hex", the bytes are encoded as a hexidecimalized string. The "esc"
	// encoding is a C-string like encoding, and is more compact than hex if most of
	// the data to be logged is text. Printable us-ascii chars are unmodified. Common
	// "C" control chars are represented as "\r", "\n", "\t", etc. Non-printable and
	// byte values greater than 0x80 are escaped using a backslash and hex encoding:
	// \xHH. Certain printable chars are backslashed: SPACE, double-quote,
	// single-quote, etc.
	// 
	void get_SessionLogEncoding(CkString &str);
	// Controls how the data is encoded in the SessionLog. Possible values are "esc"
	// and "hex". The default value is "esc".
	// 
	// When set to "hex", the bytes are encoded as a hexidecimalized string. The "esc"
	// encoding is a C-string like encoding, and is more compact than hex if most of
	// the data to be logged is text. Printable us-ascii chars are unmodified. Common
	// "C" control chars are represented as "\r", "\n", "\t", etc. Non-printable and
	// byte values greater than 0x80 are escaped using a backslash and hex encoding:
	// \xHH. Certain printable chars are backslashed: SPACE, double-quote,
	// single-quote, etc.
	// 
	const char *sessionLogEncoding(void);
	// Controls how the data is encoded in the SessionLog. Possible values are "esc"
	// and "hex". The default value is "esc".
	// 
	// When set to "hex", the bytes are encoded as a hexidecimalized string. The "esc"
	// encoding is a C-string like encoding, and is more compact than hex if most of
	// the data to be logged is text. Printable us-ascii chars are unmodified. Common
	// "C" control chars are represented as "\r", "\n", "\t", etc. Non-printable and
	// byte values greater than 0x80 are escaped using a backslash and hex encoding:
	// \xHH. Certain printable chars are backslashed: SPACE, double-quote,
	// single-quote, etc.
	// 
	void put_SessionLogEncoding(const char *newVal);

	// Specifies the SNI hostname to be used in the TLS ClientHello. This property is
	// only needed when the domain is specified via a dotted IP address and an SNI
	// hostname is desired. (Normally, Chilkat automatically uses the domain name in
	// the SNI hostname TLS ClientHello extension.)
	void get_SniHostname(CkString &str);
	// Specifies the SNI hostname to be used in the TLS ClientHello. This property is
	// only needed when the domain is specified via a dotted IP address and an SNI
	// hostname is desired. (Normally, Chilkat automatically uses the domain name in
	// the SNI hostname TLS ClientHello extension.)
	const char *sniHostname(void);
	// Specifies the SNI hostname to be used in the TLS ClientHello. This property is
	// only needed when the domain is specified via a dotted IP address and an SNI
	// hostname is desired. (Normally, Chilkat automatically uses the domain name in
	// the SNI hostname TLS ClientHello extension.)
	void put_SniHostname(const char *newVal);

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
	int get_SoRcvBuf(void);
	// Sets the receive buffer size socket option. Normally, this property should be
	// left unchanged. The default value is 4194304.
	// 
	// This property can be increased if download performance seems slow. It is
	// recommended to be a multiple of 4096.
	// 
	void put_SoRcvBuf(int newVal);

	// Sets the SO_REUSEADDR socket option for a socket that will bind to a port and
	// listen for incoming connections. The default value is true, meaning that the
	// SO_REUSEADDR socket option is set. If the socket option must be unset, set this
	// property equal to false prior to calling BindAndListen or InitSslServer.
	bool get_SoReuseAddr(void);
	// Sets the SO_REUSEADDR socket option for a socket that will bind to a port and
	// listen for incoming connections. The default value is true, meaning that the
	// SO_REUSEADDR socket option is set. If the socket option must be unset, set this
	// property equal to false prior to calling BindAndListen or InitSslServer.
	void put_SoReuseAddr(bool newVal);

	// Sets the send buffer size socket option. Normally, this property should be left
	// unchanged. The default value is 262144.
	// 
	// This property can be increased if upload performance seems slow. It is
	// recommended to be a multiple of 4096. Testing with sizes such as 512K and 1MB is
	// reasonable.
	// 
	int get_SoSndBuf(void);
	// Sets the send buffer size socket option. Normally, this property should be left
	// unchanged. The default value is 262144.
	// 
	// This property can be increased if upload performance seems slow. It is
	// recommended to be a multiple of 4096. Testing with sizes such as 512K and 1MB is
	// reasonable.
	// 
	void put_SoSndBuf(int newVal);

	// Set this property to true if SSL/TLS is required for accepted connections
	// (AcceptNextConnection). The default value is false.
	// 
	// Note: This property should have been more precisely named "RequireSslClient". It
	// is a property that if set to true, requires all accepted connections use
	// SSL/TLS. If a client attempts to connect but cannot establish the TLS
	// connection, then it is not accepted. This property is not meant to reflect the
	// current state of the connection.
	// 
	// The TlsVersion property shows the current or last negotiated TLS version of the
	// connection. The TlsVersion will be empty for a non-SSL/TLS connection.
	// 
	bool get_Ssl(void);
	// Set this property to true if SSL/TLS is required for accepted connections
	// (AcceptNextConnection). The default value is false.
	// 
	// Note: This property should have been more precisely named "RequireSslClient". It
	// is a property that if set to true, requires all accepted connections use
	// SSL/TLS. If a client attempts to connect but cannot establish the TLS
	// connection, then it is not accepted. This property is not meant to reflect the
	// current state of the connection.
	// 
	// The TlsVersion property shows the current or last negotiated TLS version of the
	// connection. The TlsVersion will be empty for a non-SSL/TLS connection.
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

	// A charset such as "utf-8", "windows-1252", "Shift_JIS", "iso-8859-1", etc.
	// Methods for sending and receiving strings will use this charset as the encoding.
	// Strings sent on the socket are first converted (if necessary) to this encoding.
	// When reading, it is assumed that the bytes received are converted FROM this
	// charset if necessary. This ONLY APPLIES TO THE SendString and ReceiveString
	// methods. The default value is "ansi".
	void get_StringCharset(CkString &str);
	// A charset such as "utf-8", "windows-1252", "Shift_JIS", "iso-8859-1", etc.
	// Methods for sending and receiving strings will use this charset as the encoding.
	// Strings sent on the socket are first converted (if necessary) to this encoding.
	// When reading, it is assumed that the bytes received are converted FROM this
	// charset if necessary. This ONLY APPLIES TO THE SendString and ReceiveString
	// methods. The default value is "ansi".
	const char *stringCharset(void);
	// A charset such as "utf-8", "windows-1252", "Shift_JIS", "iso-8859-1", etc.
	// Methods for sending and receiving strings will use this charset as the encoding.
	// Strings sent on the socket are first converted (if necessary) to this encoding.
	// When reading, it is assumed that the bytes received are converted FROM this
	// charset if necessary. This ONLY APPLIES TO THE SendString and ReceiveString
	// methods. The default value is "ansi".
	void put_StringCharset(const char *newVal);

	// Controls whether the TCP_NODELAY socket option is used for the underlying TCP/IP
	// socket. The default value is false. Setting the value to true disables the
	// Nagle algorithm and allows for better performance when small amounts of data are
	// sent on the socket connection.
	bool get_TcpNoDelay(void);
	// Controls whether the TCP_NODELAY socket option is used for the underlying TCP/IP
	// socket. The default value is false. Setting the value to true disables the
	// Nagle algorithm and allows for better performance when small amounts of data are
	// sent on the socket connection.
	void put_TcpNoDelay(bool newVal);

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
	// defaults to the empty string and should typically remain empty. Can be set to a
	// list of the following comma separated keywords:
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "EnableTls13" - Introduced in v9.5.0.82. Causes TLS 1.3 to be offered in the
	//     ClientHello of the TLS protocol, allowing the server to select TLS 1.3 for the
	//     session. Future versions of Chilkat will enable TLS 1.3 by default. This option
	//     is only necessary in v9.5.0.82 if TLS 1.3 is desired.
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty. Can be set to a
	// list of the following comma separated keywords:
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "EnableTls13" - Introduced in v9.5.0.82. Causes TLS 1.3 to be offered in the
	//     ClientHello of the TLS protocol, allowing the server to select TLS 1.3 for the
	//     session. Future versions of Chilkat will enable TLS 1.3 by default. This option
	//     is only necessary in v9.5.0.82 if TLS 1.3 is desired.
	const char *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty. Can be set to a
	// list of the following comma separated keywords:
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "EnableTls13" - Introduced in v9.5.0.82. Causes TLS 1.3 to be offered in the
	//     ClientHello of the TLS protocol, allowing the server to select TLS 1.3 for the
	//     session. Future versions of Chilkat will enable TLS 1.3 by default. This option
	//     is only necessary in v9.5.0.82 if TLS 1.3 is desired.
	void put_UncommonOptions(const char *newVal);

	// Provides a way to store text data with the socket object. The UserData is purely
	// for convenience and is not involved in the socket communications in any way. An
	// application might use this property to keep extra information associated with
	// the socket.
	void get_UserData(CkString &str);
	// Provides a way to store text data with the socket object. The UserData is purely
	// for convenience and is not involved in the socket communications in any way. An
	// application might use this property to keep extra information associated with
	// the socket.
	const char *userData(void);
	// Provides a way to store text data with the socket object. The UserData is purely
	// for convenience and is not involved in the socket communications in any way. An
	// application might use this property to keep extra information associated with
	// the socket.
	void put_UserData(const char *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Blocking call to accept the next incoming connection on the socket. maxWaitMs
	// specifies the maximum time to wait (in milliseconds). Set this to 0 to wait
	// indefinitely. If successful, a new socket object is returned.
	// 
	// Important: If accepting an SSL/TLS connection, the SSL handshake is part of the
	// connection establishment process. This involves a few back-and-forth messages
	// between the client and server to establish algorithms and a shared key to create
	// the secure channel. The sending and receiving of these messages are governed by
	// the MaxReadIdleMs and MaxSendIdleMs properties. If these properties are set to 0
	// (and this is the default unless changed by your application), then the
	// AcceptNextConnection can hang indefinitely during the SSL handshake process.
	// Make sure these properties are set to appropriate values before calling this
	// method.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkSocket *AcceptNextConnection(int maxWaitMs);

	// Blocking call to accept the next incoming connection on the socket. maxWaitMs
	// specifies the maximum time to wait (in milliseconds). Set this to 0 to wait
	// indefinitely. If successful, a new socket object is returned.
	// 
	// Important: If accepting an SSL/TLS connection, the SSL handshake is part of the
	// connection establishment process. This involves a few back-and-forth messages
	// between the client and server to establish algorithms and a shared key to create
	// the secure channel. The sending and receiving of these messages are governed by
	// the MaxReadIdleMs and MaxSendIdleMs properties. If these properties are set to 0
	// (and this is the default unless changed by your application), then the
	// AcceptNextConnection can hang indefinitely during the SSL handshake process.
	// Make sure these properties are set to appropriate values before calling this
	// method.
	// 
	CkTask *AcceptNextConnectionAsync(int maxWaitMs);


	// If this object is a server-side socket accepting SSL/TLS connections, and wishes
	// to require a client-side certificate for authentication, then it should make one
	// or more calls to this method to identify the CA's it will accept for client-side
	// certificates.
	// 
	// If no CA DN's are added by this method, then client certificates from any root
	// CA are accepted.
	// 
	// Important: If calling this method, it must be called before calling
	// InitSslServer.
	// 
	bool AddSslAcceptableClientCaDn(const char *certAuthDN);


	// Binds a TCP socket to a port and configures it to listen for incoming
	// connections. The size of the backlog is passed in backLog. The backLog is necessary
	// when multiple connections arrive at the same time, or close enough in time such
	// that they cannot be serviced immediately. (A typical value to use for backLog is
	// 5.) This method should be called once prior to receiving incoming connection
	// requests via the AcceptNextConnection or AsyncAcceptStart methods.
	// 
	// Note:This method will find a random unused port to listen on if you bind to port
	// 0. The chosen port is available via the read-only ListenPort property after this
	// method returns successful.
	// 
	// To bind and listen using IPv6, set the ListenIpv6 property = true prior to
	// calling this method.
	// 
	// What is a reasonable value for backLog? The answer depends on how many simultaneous
	// incoming connections could be expected, and how quickly your application can
	// process an incoming connection and then return to accept the next connection.
	// 
	bool BindAndListen(int port, int backLog);

	// Binds a TCP socket to a port and configures it to listen for incoming
	// connections. The size of the backlog is passed in backLog. The backLog is necessary
	// when multiple connections arrive at the same time, or close enough in time such
	// that they cannot be serviced immediately. (A typical value to use for backLog is
	// 5.) This method should be called once prior to receiving incoming connection
	// requests via the AcceptNextConnection or AsyncAcceptStart methods.
	// 
	// Note:This method will find a random unused port to listen on if you bind to port
	// 0. The chosen port is available via the read-only ListenPort property after this
	// method returns successful.
	// 
	// To bind and listen using IPv6, set the ListenIpv6 property = true prior to
	// calling this method.
	// 
	// What is a reasonable value for backLog? The answer depends on how many simultaneous
	// incoming connections could be expected, and how quickly your application can
	// process an incoming connection and then return to accept the next connection.
	// 
	CkTask *BindAndListenAsync(int port, int backLog);


	// Binds a TCP socket to an unused port within a port range (beginPort to endPort) and
	// configures it to listen for incoming connections. The size of the backlog is
	// passed in endPort. The endPort is necessary when multiple connections arrive at the
	// same time, or close enough in time such that they cannot be serviced
	// immediately. (A typical value to use for endPort is 5.) This method should be
	// called once prior to receiving incoming connection requests via the
	// AcceptNextConnection method.
	// 
	// To bind and listen using IPv6, set the ListenIpv6 property = true prior to
	// calling this method.
	// 
	// Returns the port number that was bound, or -1 if no port was available or if it
	// failed for some other reason.
	// 
	int BindAndListenPortRange(int beginPort, int endPort, int backLog);

	// Binds a TCP socket to an unused port within a port range (beginPort to endPort) and
	// configures it to listen for incoming connections. The size of the backlog is
	// passed in endPort. The endPort is necessary when multiple connections arrive at the
	// same time, or close enough in time such that they cannot be serviced
	// immediately. (A typical value to use for endPort is 5.) This method should be
	// called once prior to receiving incoming connection requests via the
	// AcceptNextConnection method.
	// 
	// To bind and listen using IPv6, set the ListenIpv6 property = true prior to
	// calling this method.
	// 
	// Returns the port number that was bound, or -1 if no port was available or if it
	// failed for some other reason.
	// 
	CkTask *BindAndListenPortRangeAsync(int beginPort, int endPort, int backLog);


	// Convenience method for building a simple HTTP GET request from a URL.
	bool BuildHttpGetRequest(const char *url, CkString &outStr);

	// Convenience method for building a simple HTTP GET request from a URL.
	const char *buildHttpGetRequest(const char *url);

	// Determines if the socket is writeable. Returns one of the following integer
	// values:
	// 
	// 1: If the socket is connected and ready for writing.
	// 0: If a timeout occurred or if the application aborted the method during an
	// event callback.
	// -1: The socket is not connected.
	// 
	// A maxWaitMs value of 0 indicates a poll.
	// 
	int CheckWriteable(int maxWaitMs);

	// Determines if the socket is writeable. Returns one of the following integer
	// values:
	// 
	// 1: If the socket is connected and ready for writing.
	// 0: If a timeout occurred or if the application aborted the method during an
	// event callback.
	// -1: The socket is not connected.
	// 
	// A maxWaitMs value of 0 indicates a poll.
	// 
	CkTask *CheckWriteableAsync(int maxWaitMs);


	// Clears the contents of the SessionLog property.
	void ClearSessionLog(void);


	// Creates a copy that shares the same underlying TCP (or SSL/TLS) connection. This
	// allows for simultaneous reading/writing by different threads on the socket. When
	// using asynchronous reading/writing, it is not necessary to clone the socket.
	// However, if separate background threads are making synchronous calls to
	// read/write, then one thread may use the original socket, and the other should
	// use a clone.
	// The caller is responsible for deleting the object returned by this method.
	CkSocket *CloneSocket(void);


	// Cleanly terminates and closes a TCP, TLS, or SSH channel connection. The maxWaitMs
	// applies to SSL/TLS connections because there is a handshake that occurs during
	// secure channel shutdown.
	bool Close(int maxWaitMs);

	// Cleanly terminates and closes a TCP, TLS, or SSH channel connection. The maxWaitMs
	// applies to SSL/TLS connections because there is a handshake that occurs during
	// secure channel shutdown.
	CkTask *CloseAsync(int maxWaitMs);


	// Establishes a secure SSL/TLS or a plain non-secure TCP connection with a remote
	// host:port. This is a blocking call. The maximum wait time (in milliseconds) is
	// passed in maxWaitMs. This is the amount of time the app is willing to wait for the
	// TCP connection to be accepted.
	// 
	// To establish an SSL/TLS connection, set ssl = true, otherwise set ssl =
	// false for a normal TCP connection. Note: The timeouts that apply to the
	// internal SSL/TLS handshaking messages are the MaxReadIdleMs and MaxSendIdleMs
	// properties.
	// 
	// Note: Connections do not automatically close because of inactivity. A connection
	// will remain open indefinitely even if there is no activity.
	// 
	// Important: All TCP-based Internet communications, regardless of the protocol
	// (such as HTTP, FTP, SSH, IMAP, POP3, SMTP, etc.), and regardless of SSL/TLS,
	// begin with establishing a TCP connection to a remote host:port. External
	// security-related infrastructure such as software firewalls (Windows Firewall),
	// hardware firewalls, anti-virus, at either source or destination (or both) can
	// block the connection. If the connection fails, make sure to check all potential
	// external causes of blockage.
	// 
	// Question: How do I Choose the TLS version, such as 1.2? Answer: The client does
	// not specifically choose the TLS version. In the TLS handshake (which is what
	// occurs internally in this method), the client tells the server the version of
	// the TLS protocol it wishes to use, which should be the highest version is
	// supports. In this case, (at the time of this writing on 22-June-2017) it is TLS
	// 1.2. The server then chooses the TLS version that will actually be used. In most
	// cases it will be TLS 1.2. The client can then choose to accept or reject the
	// connection based on the TLS version chosen by the server. By default, Chilkat
	// will reject anything lower than SSL 3.0 (i.e. SSL 2.0 or lower is rejected). The
	// SslProtocol property can be set to change what is accepted by Chilkat. For
	// example, it can be set to "TLS 1.0 or higher".
	// 
	bool Connect(const char *hostname, int port, bool ssl, int maxWaitMs);

	// Establishes a secure SSL/TLS or a plain non-secure TCP connection with a remote
	// host:port. This is a blocking call. The maximum wait time (in milliseconds) is
	// passed in maxWaitMs. This is the amount of time the app is willing to wait for the
	// TCP connection to be accepted.
	// 
	// To establish an SSL/TLS connection, set ssl = true, otherwise set ssl =
	// false for a normal TCP connection. Note: The timeouts that apply to the
	// internal SSL/TLS handshaking messages are the MaxReadIdleMs and MaxSendIdleMs
	// properties.
	// 
	// Note: Connections do not automatically close because of inactivity. A connection
	// will remain open indefinitely even if there is no activity.
	// 
	// Important: All TCP-based Internet communications, regardless of the protocol
	// (such as HTTP, FTP, SSH, IMAP, POP3, SMTP, etc.), and regardless of SSL/TLS,
	// begin with establishing a TCP connection to a remote host:port. External
	// security-related infrastructure such as software firewalls (Windows Firewall),
	// hardware firewalls, anti-virus, at either source or destination (or both) can
	// block the connection. If the connection fails, make sure to check all potential
	// external causes of blockage.
	// 
	// Question: How do I Choose the TLS version, such as 1.2? Answer: The client does
	// not specifically choose the TLS version. In the TLS handshake (which is what
	// occurs internally in this method), the client tells the server the version of
	// the TLS protocol it wishes to use, which should be the highest version is
	// supports. In this case, (at the time of this writing on 22-June-2017) it is TLS
	// 1.2. The server then chooses the TLS version that will actually be used. In most
	// cases it will be TLS 1.2. The client can then choose to accept or reject the
	// connection based on the TLS version chosen by the server. By default, Chilkat
	// will reject anything lower than SSL 3.0 (i.e. SSL 2.0 or lower is rejected). The
	// SslProtocol property can be set to change what is accepted by Chilkat. For
	// example, it can be set to "TLS 1.0 or higher".
	// 
	CkTask *ConnectAsync(const char *hostname, int port, bool ssl, int maxWaitMs);


	// Closes the secure (TLS/SSL) channel leaving the socket in a connected state
	// where data sent and received is unencrypted.
	bool ConvertFromSsl(void);

	// Closes the secure (TLS/SSL) channel leaving the socket in a connected state
	// where data sent and received is unencrypted.
	CkTask *ConvertFromSslAsync(void);


	// Converts a non-SSL/TLS connected socket to a secure channel using TLS/SSL.
	bool ConvertToSsl(void);

	// Converts a non-SSL/TLS connected socket to a secure channel using TLS/SSL.
	CkTask *ConvertToSslAsync(void);


	// Clears the Chilkat-wide in-memory hostname-to-IP address DNS cache. Chilkat
	// automatically maintains this in-memory cache to prevent redundant DNS lookups.
	// If the TTL on the DNS A records being accessed are short and/or these DNS
	// records change frequently, then this method can be called clear the internal
	// cache. Note: The DNS cache is used/shared among all Chilkat objects in a
	// program, and clearing the cache affects all Chilkat objects.
	void DnsCacheClear(void);


	// Performs a DNS query to resolve a hostname to an IP address. The IP address is
	// returned if successful. The maximum time to wait (in milliseconds) is passed in
	// maxWaitMs. To wait indefinitely, set maxWaitMs = 0.
	bool DnsLookup(const char *hostname, int maxWaitMs, CkString &outStr);

	// Performs a DNS query to resolve a hostname to an IP address. The IP address is
	// returned if successful. The maximum time to wait (in milliseconds) is passed in
	// maxWaitMs. To wait indefinitely, set maxWaitMs = 0.
	const char *dnsLookup(const char *hostname, int maxWaitMs);
	// Performs a DNS query to resolve a hostname to an IP address. The IP address is
	// returned if successful. The maximum time to wait (in milliseconds) is passed in
	// maxWaitMs. To wait indefinitely, set maxWaitMs = 0.
	CkTask *DnsLookupAsync(const char *hostname, int maxWaitMs);


	// Returns the digital certificate to be used for SSL connections. This method
	// would only be called by an SSL server application. The SSL certificate is
	// initially specified by calling InitSslServer.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetMyCert(void);


	// Returns the Nth client certificate received during an SSL/TLS handshake. This
	// method only applies to the server-side of an SSL/TLS connection. The 1st client
	// certificate is at index 0. The NumReceivedClientCerts property indicates the
	// number of client certificates received during the SSL/TLS connection
	// establishment.
	// 
	// Client certificates are customarily only sent when the server demands
	// client-side authentication, as in 2-way SSL/TLS. This method provides the
	// ability for the server to access and examine the client-side certs immediately
	// after a connection is established. (Of course, if the client-side certs are
	// inadequate for authentication, then the application can choose to immediately
	// disconnect.)
	// 
	// Important: This method should be called from the socket object that is returned
	// by AcceptNextConnection.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetReceivedClientCert(int index);


	// If connected as an SSL/TLS client to an SSL/TLS server where the server requires
	// a client-side certificate for authentication, then the NumSslAcceptableClientCAs
	// property contains the number of acceptable certificate authorities sent by the
	// server during connection establishment handshake. This method may be called to
	// get the Distinguished Name (DN) of each acceptable CA. The index should range
	// from 0 to NumSslAcceptableClientCAs - 1.
	bool GetSslAcceptableClientCaDn(int index, CkString &outStr);

	// If connected as an SSL/TLS client to an SSL/TLS server where the server requires
	// a client-side certificate for authentication, then the NumSslAcceptableClientCAs
	// property contains the number of acceptable certificate authorities sent by the
	// server during connection establishment handshake. This method may be called to
	// get the Distinguished Name (DN) of each acceptable CA. The index should range
	// from 0 to NumSslAcceptableClientCAs - 1.
	const char *getSslAcceptableClientCaDn(int index);
	// If connected as an SSL/TLS client to an SSL/TLS server where the server requires
	// a client-side certificate for authentication, then the NumSslAcceptableClientCAs
	// property contains the number of acceptable certificate authorities sent by the
	// server during connection establishment handshake. This method may be called to
	// get the Distinguished Name (DN) of each acceptable CA. The index should range
	// from 0 to NumSslAcceptableClientCAs - 1.
	const char *sslAcceptableClientCaDn(int index);


	// Returns the SSL server's digital certificate. This method would only be called
	// by the client-side of an SSL connection. It returns the certificate of the
	// remote SSL server for the current SSL connection. If the socket is not
	// connected, or is not connected via SSL, then a NULL reference is returned.
	// The caller is responsible for deleting the object returned by this method.
	CkCert *GetSslServerCert(void);


	// SSL Server applications should call this method with the SSL server certificate
	// to be used for SSL connections. It should be called prior to accepting
	// connections. This method has an intended side-effect: If not already connected,
	// then the Ssl property is set to true.
	bool InitSslServer(CkCert &cert);


	// Returns true if the component is unlocked.
	bool IsUnlocked(void);


	// Provides information about what transpired in the last method called on this
	// object instance. For many methods, there is no information. However, for some
	// methods, details about what occurred can be obtained by getting the LastJsonData
	// right after the method call returns.
	// The caller is responsible for deleting the object returned by this method.
	CkJsonObject *LastJsonData(void);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Loads the socket object from a completed asynchronous task.
	bool LoadTaskResult(CkTask &task);


	// Check to see if data is available for reading on the socket. Returns true if
	// data is waiting and false if no data is waiting to be read.
	bool PollDataAvailable(void);

	// Check to see if data is available for reading on the socket. Returns true if
	// data is waiting and false if no data is waiting to be read.
	CkTask *PollDataAvailableAsync(void);


	// Receives as much data as is immediately available on a connected TCP socket and
	// appends the incoming data to binData. If no data is immediately available, it waits
	// up to MaxReadIdleMs milliseconds for data to arrive.
	bool ReceiveBd(CkBinData &binData);

	// Receives as much data as is immediately available on a connected TCP socket and
	// appends the incoming data to binData. If no data is immediately available, it waits
	// up to MaxReadIdleMs milliseconds for data to arrive.
	CkTask *ReceiveBdAsync(CkBinData &binData);


	// Reads exactly numBytes bytes from the connection. This method blocks until numBytes
	// bytes are read or the read times out. The timeout is specified by the
	// MaxReadIdleMs property (in milliseconds).
	bool ReceiveBdN(unsigned long numBytes, CkBinData &binData);

	// Reads exactly numBytes bytes from the connection. This method blocks until numBytes
	// bytes are read or the read times out. The timeout is specified by the
	// MaxReadIdleMs property (in milliseconds).
	CkTask *ReceiveBdNAsync(unsigned long numBytes, CkBinData &binData);


	// Receives a single byte. The received byte will be available in the ReceivedInt
	// property. If bUnsigned is true, then a value from 0 to 255 is returned in
	// ReceivedInt. If bUnsigned is false, then a value from -128 to +127 is returned.
	bool ReceiveByte(bool bUnsigned);

	// Receives a single byte. The received byte will be available in the ReceivedInt
	// property. If bUnsigned is true, then a value from 0 to 255 is returned in
	// ReceivedInt. If bUnsigned is false, then a value from -128 to +127 is returned.
	CkTask *ReceiveByteAsync(bool bUnsigned);


	// Receives as much data as is immediately available on a connected TCP socket. If
	// no data is immediately available, it waits up to MaxReadIdleMs milliseconds for
	// data to arrive.
	bool ReceiveBytes(CkByteData &outData);

	// Receives as much data as is immediately available on a connected TCP socket. If
	// no data is immediately available, it waits up to MaxReadIdleMs milliseconds for
	// data to arrive.
	CkTask *ReceiveBytesAsync(void);


	// The same as ReceiveBytes, except the bytes are returned in encoded string form
	// according to encodingAlg. The encodingAlg can be "Base64", "modBase64", "Base32", "UU", "QP"
	// (for quoted-printable), "URL" (for url-encoding), "Hex", "Q", "B", "url_oath",
	// "url_rfc1738", "url_rfc2396", or "url_rfc3986".
	bool ReceiveBytesENC(const char *encodingAlg, CkString &outStr);

	// The same as ReceiveBytes, except the bytes are returned in encoded string form
	// according to encodingAlg. The encodingAlg can be "Base64", "modBase64", "Base32", "UU", "QP"
	// (for quoted-printable), "URL" (for url-encoding), "Hex", "Q", "B", "url_oath",
	// "url_rfc1738", "url_rfc2396", or "url_rfc3986".
	const char *receiveBytesENC(const char *encodingAlg);
	// The same as ReceiveBytes, except the bytes are returned in encoded string form
	// according to encodingAlg. The encodingAlg can be "Base64", "modBase64", "Base32", "UU", "QP"
	// (for quoted-printable), "URL" (for url-encoding), "Hex", "Q", "B", "url_oath",
	// "url_rfc1738", "url_rfc2396", or "url_rfc3986".
	CkTask *ReceiveBytesENCAsync(const char *encodingAlg);


	// Reads exactly numBytes bytes from a connected SSL or non-SSL socket. This method
	// blocks until numBytes bytes are read or the read times out. The timeout is specified
	// by the MaxReadIdleMs property (in milliseconds).
	bool ReceiveBytesN(unsigned long numBytes, CkByteData &outData);

	// Reads exactly numBytes bytes from a connected SSL or non-SSL socket. This method
	// blocks until numBytes bytes are read or the read times out. The timeout is specified
	// by the MaxReadIdleMs property (in milliseconds).
	CkTask *ReceiveBytesNAsync(unsigned long numBytes);


	// Receives as much data as is immediately available on a connected TCP socket. If
	// no data is immediately available, it waits up to MaxReadIdleMs milliseconds for
	// data to arrive.
	// 
	// The received data is appended to the file specified by appendFilename.
	// 
	bool ReceiveBytesToFile(const char *appendFilename);

	// Receives as much data as is immediately available on a connected TCP socket. If
	// no data is immediately available, it waits up to MaxReadIdleMs milliseconds for
	// data to arrive.
	// 
	// The received data is appended to the file specified by appendFilename.
	// 
	CkTask *ReceiveBytesToFileAsync(const char *appendFilename);


	// Receives a 4-byte signed integer and returns the value received. Returns -1 on
	// error.
	int ReceiveCount(void);

	// Receives a 4-byte signed integer and returns the value received. Returns -1 on
	// error.
	CkTask *ReceiveCountAsync(void);


	// Receives a 16-bit integer (2 bytes). The received integer will be available in
	// the ReceivedInt property. Set bigEndian equal to true if the incoming 16-bit
	// integer is in big-endian byte order. Otherwise set bigEndian equal to false for
	// receving a little-endian integer. If bUnsigned is true, the ReceivedInt will range
	// from 0 to 65,535. If bUnsigned is false, the ReceivedInt will range from -32,768
	// through 32,767.
	bool ReceiveInt16(bool bigEndian, bool bUnsigned);

	// Receives a 16-bit integer (2 bytes). The received integer will be available in
	// the ReceivedInt property. Set bigEndian equal to true if the incoming 16-bit
	// integer is in big-endian byte order. Otherwise set bigEndian equal to false for
	// receving a little-endian integer. If bUnsigned is true, the ReceivedInt will range
	// from 0 to 65,535. If bUnsigned is false, the ReceivedInt will range from -32,768
	// through 32,767.
	CkTask *ReceiveInt16Async(bool bigEndian, bool bUnsigned);


	// Receives a 32-bit integer (4 bytes). The received integer will be available in
	// the ReceivedInt property. Set bigEndian equal to true if the incoming 32-bit
	// integer is in big-endian byte order. Otherwise set bigEndian equal to false for
	// receving a little-endian integer.
	bool ReceiveInt32(bool bigEndian);

	// Receives a 32-bit integer (4 bytes). The received integer will be available in
	// the ReceivedInt property. Set bigEndian equal to true if the incoming 32-bit
	// integer is in big-endian byte order. Otherwise set bigEndian equal to false for
	// receving a little-endian integer.
	CkTask *ReceiveInt32Async(bool bigEndian);


	// The same as ReceiveBytesN, except the bytes are returned in encoded string form
	// using the encoding specified by numBytes. The numBytes can be "Base64", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Hex",
	// "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", or "url_rfc3986".
	bool ReceiveNBytesENC(unsigned long numBytes, const char *encodingAlg, CkString &outStr);

	// The same as ReceiveBytesN, except the bytes are returned in encoded string form
	// using the encoding specified by numBytes. The numBytes can be "Base64", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Hex",
	// "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", or "url_rfc3986".
	const char *receiveNBytesENC(unsigned long numBytes, const char *encodingAlg);
	// The same as ReceiveBytesN, except the bytes are returned in encoded string form
	// using the encoding specified by numBytes. The numBytes can be "Base64", "modBase64",
	// "Base32", "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Hex",
	// "Q", "B", "url_oath", "url_rfc1738", "url_rfc2396", or "url_rfc3986".
	CkTask *ReceiveNBytesENCAsync(unsigned long numBytes, const char *encodingAlg);


	// Receives as much data as is immediately available on the connection. If no data
	// is immediately available, it waits up to MaxReadIdleMs milliseconds for data to
	// arrive. The incoming bytes are interpreted according to the StringCharset
	// property and appended to sb.
	bool ReceiveSb(CkStringBuilder &sb);

	// Receives as much data as is immediately available on the connection. If no data
	// is immediately available, it waits up to MaxReadIdleMs milliseconds for data to
	// arrive. The incoming bytes are interpreted according to the StringCharset
	// property and appended to sb.
	CkTask *ReceiveSbAsync(CkStringBuilder &sb);


	// Receives as much data as is immediately available on a TCP/IP or SSL socket. If
	// no data is immediately available, it waits up to MaxReadIdleMs milliseconds for
	// data to arrive. The incoming bytes are interpreted according to the
	// StringCharset property and returned as a string.
	bool ReceiveString(CkString &outStr);

	// Receives as much data as is immediately available on a TCP/IP or SSL socket. If
	// no data is immediately available, it waits up to MaxReadIdleMs milliseconds for
	// data to arrive. The incoming bytes are interpreted according to the
	// StringCharset property and returned as a string.
	const char *receiveString(void);
	// Receives as much data as is immediately available on a TCP/IP or SSL socket. If
	// no data is immediately available, it waits up to MaxReadIdleMs milliseconds for
	// data to arrive. The incoming bytes are interpreted according to the
	// StringCharset property and returned as a string.
	CkTask *ReceiveStringAsync(void);


	// Same as ReceiveString, but limits the amount of data returned to a maximum of
	// maxByteCount bytes.
	// 
	// (Receives as much data as is immediately available on the TCP/IP or SSL socket.
	// If no data is immediately available, it waits up to MaxReadIdleMs milliseconds
	// for data to arrive. The incoming bytes are interpreted according to the
	// StringCharset property and returned as a string.)
	// 
	bool ReceiveStringMaxN(int maxByteCount, CkString &outStr);

	// Same as ReceiveString, but limits the amount of data returned to a maximum of
	// maxByteCount bytes.
	// 
	// (Receives as much data as is immediately available on the TCP/IP or SSL socket.
	// If no data is immediately available, it waits up to MaxReadIdleMs milliseconds
	// for data to arrive. The incoming bytes are interpreted according to the
	// StringCharset property and returned as a string.)
	// 
	const char *receiveStringMaxN(int maxByteCount);
	// Same as ReceiveString, but limits the amount of data returned to a maximum of
	// maxByteCount bytes.
	// 
	// (Receives as much data as is immediately available on the TCP/IP or SSL socket.
	// If no data is immediately available, it waits up to MaxReadIdleMs milliseconds
	// for data to arrive. The incoming bytes are interpreted according to the
	// StringCharset property and returned as a string.)
	// 
	CkTask *ReceiveStringMaxNAsync(int maxByteCount);


	// Receives bytes on a connected SSL or non-SSL socket until a specific 1-byte
	// value is read. Returns a string containing all the bytes up to but excluding the
	// lookForByte.
	bool ReceiveStringUntilByte(int lookForByte, CkString &outStr);

	// Receives bytes on a connected SSL or non-SSL socket until a specific 1-byte
	// value is read. Returns a string containing all the bytes up to but excluding the
	// lookForByte.
	const char *receiveStringUntilByte(int lookForByte);
	// Receives bytes on a connected SSL or non-SSL socket until a specific 1-byte
	// value is read. Returns a string containing all the bytes up to but excluding the
	// lookForByte.
	CkTask *ReceiveStringUntilByteAsync(int lookForByte);


	// Reads text from the connected TCP/IP or SSL socket until a CRLF is received.
	// Returns the text up to and including the CRLF. The incoming bytes are
	// interpreted according to the charset specified by the StringCharset property.
	bool ReceiveToCRLF(CkString &outStr);

	// Reads text from the connected TCP/IP or SSL socket until a CRLF is received.
	// Returns the text up to and including the CRLF. The incoming bytes are
	// interpreted according to the charset specified by the StringCharset property.
	const char *receiveToCRLF(void);
	// Reads text from the connected TCP/IP or SSL socket until a CRLF is received.
	// Returns the text up to and including the CRLF. The incoming bytes are
	// interpreted according to the charset specified by the StringCharset property.
	CkTask *ReceiveToCRLFAsync(void);


	// Receives bytes on the TCP/IP or SSL socket until a specific 1-byte value is
	// read. Returns all the bytes up to and including the lookForByte.
	bool ReceiveUntilByte(int lookForByte, CkByteData &outBytes);

	// Receives bytes on the TCP/IP or SSL socket until a specific 1-byte value is
	// read. Returns all the bytes up to and including the lookForByte.
	CkTask *ReceiveUntilByteAsync(int lookForByte);


	// Receives bytes on the TCP/IP or SSL socket until a specific 1-byte value is
	// read. Returns all the bytes up to and including the lookForByte. The received bytes are
	// appended to bd.
	bool ReceiveUntilByteBd(int lookForByte, CkBinData &bd);

	// Receives bytes on the TCP/IP or SSL socket until a specific 1-byte value is
	// read. Returns all the bytes up to and including the lookForByte. The received bytes are
	// appended to bd.
	CkTask *ReceiveUntilByteBdAsync(int lookForByte, CkBinData &bd);


	// Reads text from the connected TCP/IP or SSL socket until a matching string
	// (matchStr) is received. Returns the text up to and including the matching string. As
	// an example, to one might read the header of an HTTP request or a MIME message by
	// reading up to the first double CRLF ("\r\n\r\n"). The incoming bytes are
	// interpreted according to the charset specified by the StringCharset property.
	bool ReceiveUntilMatch(const char *matchStr, CkString &outStr);

	// Reads text from the connected TCP/IP or SSL socket until a matching string
	// (matchStr) is received. Returns the text up to and including the matching string. As
	// an example, to one might read the header of an HTTP request or a MIME message by
	// reading up to the first double CRLF ("\r\n\r\n"). The incoming bytes are
	// interpreted according to the charset specified by the StringCharset property.
	const char *receiveUntilMatch(const char *matchStr);
	// Reads text from the connected TCP/IP or SSL socket until a matching string
	// (matchStr) is received. Returns the text up to and including the matching string. As
	// an example, to one might read the header of an HTTP request or a MIME message by
	// reading up to the first double CRLF ("\r\n\r\n"). The incoming bytes are
	// interpreted according to the charset specified by the StringCharset property.
	CkTask *ReceiveUntilMatchAsync(const char *matchStr);


	// Resets the performance measurements for either receiving or sending. If rcvPerf is
	// true, then the receive performance monitoring is reset. If rcvPerf is false,
	// then the sending performance monitoring is reset.
	void ResetPerf(bool rcvPerf);


	// Wait for data to arrive on this socket, or any of the contained sockets if the
	// caller is a "socket set". (If the socket is a listener socket, then waits for an
	// incoming connect. Listener sockets can be added to the "socket set" just like
	// connected sockets.)
	// 
	// (see the example at the link below for more detailed information)
	// 
	// Waits a maximum of timeoutMs milliseconds. If maxWaitMs = 0, then it is effectively a
	// poll. Returns the number of sockets with data available for reading. If no
	// sockets have data available for reading, then a value of 0 is returned. A value
	// of -1 indicates an error condition. Note: when the remote peer (in this case the
	// web server) disconnects, the socket will appear as if it has data available. A
	// "ready" socket is one where either data is available for reading or the socket
	// has become disconnected.
	// 
	// If the peer closed the connection, it will not be discovered until an attempt is
	// made to read the socket. If the read fails, then the IsConnected property may be
	// checked to see if the connection was closed.
	// 
	int SelectForReading(int timeoutMs);

	// Wait for data to arrive on this socket, or any of the contained sockets if the
	// caller is a "socket set". (If the socket is a listener socket, then waits for an
	// incoming connect. Listener sockets can be added to the "socket set" just like
	// connected sockets.)
	// 
	// (see the example at the link below for more detailed information)
	// 
	// Waits a maximum of timeoutMs milliseconds. If maxWaitMs = 0, then it is effectively a
	// poll. Returns the number of sockets with data available for reading. If no
	// sockets have data available for reading, then a value of 0 is returned. A value
	// of -1 indicates an error condition. Note: when the remote peer (in this case the
	// web server) disconnects, the socket will appear as if it has data available. A
	// "ready" socket is one where either data is available for reading or the socket
	// has become disconnected.
	// 
	// If the peer closed the connection, it will not be discovered until an attempt is
	// made to read the socket. If the read fails, then the IsConnected property may be
	// checked to see if the connection was closed.
	// 
	CkTask *SelectForReadingAsync(int timeoutMs);


	// Waits until it is known that data can be written to one or more sockets without
	// it blocking.
	// 
	// Socket writes are typically buffered by the operating system. When an
	// application writes data to a socket, the operating system appends it to the
	// socket's outgoing send buffers and returns immediately. However, if the OS send
	// buffers become filled up (because the sender is sending data faster than the
	// remote receiver can read it), then a socket write can block (until outgoing send
	// buffer space becomes available).
	// 
	// Waits a maximum of timeoutMs milliseconds. If maxWaitMs = 0, then it is effectively a
	// poll. Returns the number of sockets such that data can be written without
	// blocking. A value of -1 indicates an error condition.
	// 
	int SelectForWriting(int timeoutMs);

	// Waits until it is known that data can be written to one or more sockets without
	// it blocking.
	// 
	// Socket writes are typically buffered by the operating system. When an
	// application writes data to a socket, the operating system appends it to the
	// socket's outgoing send buffers and returns immediately. However, if the OS send
	// buffers become filled up (because the sender is sending data faster than the
	// remote receiver can read it), then a socket write can block (until outgoing send
	// buffer space becomes available).
	// 
	// Waits a maximum of timeoutMs milliseconds. If maxWaitMs = 0, then it is effectively a
	// poll. Returns the number of sockets such that data can be written without
	// blocking. A value of -1 indicates an error condition.
	// 
	CkTask *SelectForWritingAsync(int timeoutMs);


	// Sends bytes from binData over a connected SSL or non-SSL socket. If transmission
	// halts for more than MaxSendIdleMs milliseconds, the send is aborted. This is a
	// blocking (synchronous) method. It returns only after the bytes have been sent.
	// 
	// Set offset and/or numBytes to non-zero values to send a portion of the binData. If offset
	// and numBytes are both 0, then the entire binData is sent. If offset is non-zero and numBytes
	// is zero, then the bytes starting at offset until the end are sent.
	// 
	bool SendBd(CkBinData &binData, unsigned long offset, unsigned long numBytes);

	// Sends bytes from binData over a connected SSL or non-SSL socket. If transmission
	// halts for more than MaxSendIdleMs milliseconds, the send is aborted. This is a
	// blocking (synchronous) method. It returns only after the bytes have been sent.
	// 
	// Set offset and/or numBytes to non-zero values to send a portion of the binData. If offset
	// and numBytes are both 0, then the entire binData is sent. If offset is non-zero and numBytes
	// is zero, then the bytes starting at offset until the end are sent.
	// 
	CkTask *SendBdAsync(CkBinData &binData, unsigned long offset, unsigned long numBytes);


	// Sends a single byte. The integer must have a value from 0 to 255.
	bool SendByte(int value);

	// Sends a single byte. The integer must have a value from 0 to 255.
	CkTask *SendByteAsync(int value);


	// Sends bytes over a connected SSL or non-SSL socket. If transmission halts for
	// more than MaxSendIdleMs milliseconds, the send is aborted. This is a blocking
	// (synchronous) method. It returns only after the bytes have been sent.
	bool SendBytes(CkByteData &data);

	// Sends bytes over a connected SSL or non-SSL socket. If transmission halts for
	// more than MaxSendIdleMs milliseconds, the send is aborted. This is a blocking
	// (synchronous) method. It returns only after the bytes have been sent.
	CkTask *SendBytesAsync(CkByteData &data);


	// Sames as SendBytes but data is passed via a pointer and length.
	bool SendBytes2(const void *pByteData, unsigned long szByteData);


	// The same as SendBytes, except the bytes are provided in encoded string form as
	// specified by encodingAlg. The encodingAlg can be "Base64", "modBase64", "Base32", "Base58",
	// "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Hex", "Q", "B",
	// "url_oauth", "url_rfc1738", "url_rfc2396", and "url_rfc3986".
	bool SendBytesENC(const char *encodedBytes, const char *encodingAlg);

	// The same as SendBytes, except the bytes are provided in encoded string form as
	// specified by encodingAlg. The encodingAlg can be "Base64", "modBase64", "Base32", "Base58",
	// "UU", "QP" (for quoted-printable), "URL" (for url-encoding), "Hex", "Q", "B",
	// "url_oauth", "url_rfc1738", "url_rfc2396", and "url_rfc3986".
	CkTask *SendBytesENCAsync(const char *encodedBytes, const char *encodingAlg);


	// Sends a 4-byte signed integer on the connection. The receiver may call
	// ReceiveCount to receive the integer. The SendCount and ReceiveCount methods are
	// handy for sending byte counts prior to sending data. The sender would send a
	// count followed by the data, and the receiver would receive the count first, and
	// then knows how many data bytes it should expect to receive.
	bool SendCount(int byteCount);

	// Sends a 4-byte signed integer on the connection. The receiver may call
	// ReceiveCount to receive the integer. The SendCount and ReceiveCount methods are
	// handy for sending byte counts prior to sending data. The sender would send a
	// count followed by the data, and the receiver would receive the count first, and
	// then knows how many data bytes it should expect to receive.
	CkTask *SendCountAsync(int byteCount);


	// Sends a 16-bit integer (2 bytes). Set bigEndian equal to true to send the integer
	// in big-endian byte order (this is the standard network byte order). Otherwise
	// set bigEndian equal to false to send in little-endian byte order.
	bool SendInt16(int value, bool bigEndian);

	// Sends a 16-bit integer (2 bytes). Set bigEndian equal to true to send the integer
	// in big-endian byte order (this is the standard network byte order). Otherwise
	// set bigEndian equal to false to send in little-endian byte order.
	CkTask *SendInt16Async(int value, bool bigEndian);


	// Sends a 32-bit integer (4 bytes). Set bigEndian equal to true to send the integer
	// in big-endian byte order (this is the standard network byte order). Otherwise
	// set bigEndian equal to false to send in little-endian byte order.
	bool SendInt32(int value, bool bigEndian);

	// Sends a 32-bit integer (4 bytes). Set bigEndian equal to true to send the integer
	// in big-endian byte order (this is the standard network byte order). Otherwise
	// set bigEndian equal to false to send in little-endian byte order.
	CkTask *SendInt32Async(int value, bool bigEndian);


	// Sends the contents of sb over the connection. If transmission halts for more
	// than MaxSendIdleMs milliseconds, the send is aborted. The string is sent in the
	// charset encoding specified by the StringCharset property.
	// 
	// This is a blocking (synchronous) method. It returns after the string has been
	// sent.
	// 
	bool SendSb(CkStringBuilder &sb);

	// Sends the contents of sb over the connection. If transmission halts for more
	// than MaxSendIdleMs milliseconds, the send is aborted. The string is sent in the
	// charset encoding specified by the StringCharset property.
	// 
	// This is a blocking (synchronous) method. It returns after the string has been
	// sent.
	// 
	CkTask *SendSbAsync(CkStringBuilder &sb);


	// Sends a string over a connected SSL or non-SSL (TCP/IP) socket. If transmission
	// halts for more than MaxSendIdleMs milliseconds, the send is aborted. The string
	// is sent in the charset encoding specified by the StringCharset property.
	// 
	// This is a blocking (synchronous) method. It returns after the string has been
	// sent.
	// 
	bool SendString(const char *stringToSend);

	// Sends a string over a connected SSL or non-SSL (TCP/IP) socket. If transmission
	// halts for more than MaxSendIdleMs milliseconds, the send is aborted. The string
	// is sent in the charset encoding specified by the StringCharset property.
	// 
	// This is a blocking (synchronous) method. It returns after the string has been
	// sent.
	// 
	CkTask *SendStringAsync(const char *stringToSend);


	// Sends a "Wake on Lan" magic packet to a computer. A Wake on Lan is a way to
	// power on a computer remotely by sending a data packet known as a magic packet.
	// For this to work, the network card must have enabled the feature: “Power on Lan”
	// or “Power on PCI Device“, which is done by accessing the BIOS of the machine.
	// 
	// The macAddress is the MAC address (in hex) of the computer to wake. A MAC address
	// should be 6 bytes in length. For example, "000102030405". The port is the port
	// which should be 7 or 9. (Port number 9 is more commonly used.) The ipBroadcastAddr is the
	// broadcast address of your network, which usually ends with *.255. For example:
	// "192.168.1.255".
	// 
	// Your application does not call Connect prior to calling SendWakeOnLan. To use
	// this method, it's just a matter of instantiating an instance of this socket
	// object and then call SendWakeOnLan.
	// 
	bool SendWakeOnLan(const char *macAddress, int port, const char *ipBroadcastAddr);


	// The same as SendWakeOnLan, but includes an additional argument to specify a
	// SecureOn password. The password should be a hexidecimal string representing 4 or 6
	// bytes. (See https://wiki.wireshark.org/WakeOnLAN) Sending a WakeOnLAN (WOL) to
	// an IPv4 address would need a 4-byte SecureOn password, whereas an IPv6 address
	// would need a 6-byte SecureOn password.
	bool SendWakeOnLan2(const char *macAddress, int port, const char *ipBroadcastAddr, const char *password);


	// A client-side certificate for SSL/TLS connections is optional. It should be used
	// only if the server demands it. This method allows the certificate to be
	// specified using a certificate object.
	bool SetSslClientCert(CkCert &cert);


	// A client-side certificate for SSL/TLS connections is optional. It should be used
	// only if the server demands it. This method allows the certificate to be
	// specified using a PEM file.
	bool SetSslClientCertPem(const char *pemDataOrFilename, const char *pemPassword);


	// A client-side certificate for SSL/TLS connections is optional. It should be used
	// only if the server demands it. This method allows the certificate to be
	// specified using a PFX file.
	bool SetSslClientCertPfx(const char *pfxFilename, const char *pfxPassword);


	// Convenience method to force the calling thread to sleep for a number of
	// milliseconds.
	void SleepMs(int millisec);


	// Authenticates with the SSH server using public-key authentication. The
	// corresponding public key must have been installed on the SSH server for the
	// sshLogin. Authentication will succeed if the matching privateKey is provided.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	bool SshAuthenticatePk(const char *sshLogin, CkSshKey &privateKey);

	// Authenticates with the SSH server using public-key authentication. The
	// corresponding public key must have been installed on the SSH server for the
	// sshLogin. Authentication will succeed if the matching privateKey is provided.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	CkTask *SshAuthenticatePkAsync(const char *sshLogin, CkSshKey &privateKey);


	// Authenticates with the SSH server using a sshLogin and sshPassword. This method is only
	// used for SSH tunneling. The tunnel is established by calling SshOpenTunnel, then
	// (if necessary) authenticated by calling SshAuthenticatePw or SshAuthenticatePk.
	bool SshAuthenticatePw(const char *sshLogin, const char *sshPassword);

	// Authenticates with the SSH server using a sshLogin and sshPassword. This method is only
	// used for SSH tunneling. The tunnel is established by calling SshOpenTunnel, then
	// (if necessary) authenticated by calling SshAuthenticatePw or SshAuthenticatePk.
	CkTask *SshAuthenticatePwAsync(const char *sshLogin, const char *sshPassword);


	// Closes the SSH tunnel previously opened by SshOpenTunnel.
	bool SshCloseTunnel(void);

	// Closes the SSH tunnel previously opened by SshOpenTunnel.
	CkTask *SshCloseTunnelAsync(void);


	// Opens a new channel within an SSH tunnel. Returns the socket that is connected
	// to the destination host:port through the SSH tunnel via port forwarding. If ssl
	// is true, the connection is TLS (i.e. TLS inside the SSH tunnel). Returns the
	// socket object that is the port-forwarded tunneled connection. Any number of
	// channels may be opened within a single SSH tunnel, and may be port-forwarded to
	// different remote host:port endpoints.
	// The caller is responsible for deleting the object returned by this method.
	CkSocket *SshOpenChannel(const char *hostname, int port, bool ssl, int maxWaitMs);

	// Opens a new channel within an SSH tunnel. Returns the socket that is connected
	// to the destination host:port through the SSH tunnel via port forwarding. If ssl
	// is true, the connection is TLS (i.e. TLS inside the SSH tunnel). Returns the
	// socket object that is the port-forwarded tunneled connection. Any number of
	// channels may be opened within a single SSH tunnel, and may be port-forwarded to
	// different remote host:port endpoints.
	CkTask *SshOpenChannelAsync(const char *hostname, int port, bool ssl, int maxWaitMs);


	// Connects to an SSH server and creates a tunnel for port forwarding. The sshHostname is
	// the hostname (or IP address) of the SSH server. The sshPort is typically 22, which
	// is the standard SSH port number.
	// 
	// An SSH tunneling (port forwarding) session always begins by first calling
	// SshOpenTunnel to connect to the SSH server, followed by calling either
	// SshAuthenticatePw or SshAuthenticatePk to authenticate. A program would then
	// call SshOpenChannel to connect to the destination server (via the SSH tunnel).
	// Any number of channels can be opened over the same SSH tunnel.
	// 
	bool SshOpenTunnel(const char *sshHostname, int sshPort);

	// Connects to an SSH server and creates a tunnel for port forwarding. The sshHostname is
	// the hostname (or IP address) of the SSH server. The sshPort is typically 22, which
	// is the standard SSH port number.
	// 
	// An SSH tunneling (port forwarding) session always begins by first calling
	// SshOpenTunnel to connect to the SSH server, followed by calling either
	// SshAuthenticatePw or SshAuthenticatePk to authenticate. A program would then
	// call SshOpenChannel to connect to the destination server (via the SSH tunnel).
	// Any number of channels can be opened over the same SSH tunnel.
	// 
	CkTask *SshOpenTunnelAsync(const char *sshHostname, int sshPort);


	// Used in combination with the ElapsedSeconds property, which will contain the
	// number of seconds since the last call to this method. (The StartTiming method
	// and ElapsedSeconds property is provided for convenience.)
	void StartTiming(void);


	// Takes the connection from sock. If the caller of this method had an open
	// connection, then it will be closed. This method is different than the TakeSocket
	// method because the caller does not become a "socket set".
	bool TakeConnection(CkSocket &sock);


	// Takes ownership of the sock. sock is added to the internal set of connected
	// sockets. The caller object is now effectively a "socket set", i.e. a collection
	// of connected and/or listener sockets. Method calls are routed to the internal
	// sockets based on the value of the SelectorIndex property. For example, if
	// SelectorIndex equals 2, then a call to SendBytes is actually a call to SendBytes
	// on the 3rd socket in the set. (Indexing begins at 0.) Likewise, getting and
	// setting properties are also routed to the contained socket based on
	// SelectorIndex. It is possible to wait on a set of sockets for data to arrive on
	// any of them by calling SelectForReading. See the example link below.
	bool TakeSocket(CkSocket &sock);


	// Initiates a renegotiation of the TLS security parameters. This sends a
	// ClientHello to re-do the TLS handshake to establish new TLS security params.
	bool TlsRenegotiate(void);

	// Initiates a renegotiation of the TLS security parameters. This sends a
	// ClientHello to re-do the TLS handshake to establish new TLS security params.
	CkTask *TlsRenegotiateAsync(void);


	// Unlocks the component allowing for the full functionality to be used. An
	// arbitrary string can be passed to initiate a fully-functional 30-day trial.
	bool UnlockComponent(const char *unlockCode);


	// Uses an existing SSH tunnel for the connection. This is an alternative way of
	// establishing a socket connection through an SSH tunnel. There are four ways of
	// running a TCP or SSL/TLS connection through an SSH tunnel:
	//     UseSsh
	//         Establish the SSH connection and authenticate using the Chilkat SSH
	//         object.
	//         Call UseSsh to indicate that the connections should be made through the
	//         SSH tunnel.
	//         Call the Connect method to establish the TCP or SSL/TLS connection with
	//         a destination host:port. The connection is not direct, but will instead be
	//         routed through the SSH tunnel and then port-forwarded (from the SSH server) to
	//         the destination host:port. (Had UseSsh not been called, the connection would be
	//         direct.)
	//     SshOpenTunnel
	//         Call the Socket object's SshOpenTunnel method to connect to an SSH
	//         server.
	//         Call SshAuthenticatePw to authenticate with the SSH server.
	//         Instead of calling Connect to connect with the destination host:port,
	//         the SshOpenChannel method is called to connect via port-forwarding through the
	//         SSH tunnel.
	//     SshTunnel object with dynamic port forwarding
	//         The Chilkat SSH Tunnel object is utilized to run in a background thread.
	//         It connects and authenticates with an SSH server, and then listens at a port
	//         chosen by the application, and behaves as a SOCKS5 proxy server.
	//         The Socket object sets the SOCKS5 proxy host:port to
	//         localhost:_LT_port_GT_,
	//         The Socket's Connect method is called to connect via the SSH Tunnel. The
	//         connection is routed through the SSH tunnel via dynamic port forwarding.
	//         Once the background SSH Tunnel thread is running, it can handle any
	//         number of incoming connections from the foreground thread, other threads, or
	//         even other programs that are local or remote. Each incoming connection is routed
	//         via dynamic port forwarding to it's chosen destnation host:port on it's own
	//         logical SSH channel.
	//     SshTunnel object with hard-coded port forwarding
	//         The Chilkat SSH Tunnel object is utilized to run in a background thread.
	//         It connects and authenticates with an SSH server, and then listens at a port
	//         chosen by the application. It does not behave as a SOCKS5 proxy server, but
	//         instead has a hard-coded destination host:port.
	//         The Socket's Connect method is called to connect to
	//         localhost:_LT_port_GT_. The connection is automatically port-forwarded through
	//         the SSH tunnel to the hard-coded destination host:port.
	// In all cases, the SSH tunnels can hold both unencrypted TCP connections and
	// SSL/TLS connections.
	bool UseSsh(CkSsh &ssh);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
