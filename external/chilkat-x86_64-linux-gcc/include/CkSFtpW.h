// CkSFtpW.h: interface for the CkSFtpW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkSFtpW_H
#define _CkSFtpW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkByteData;
class CkTaskW;
class CkSshKeyW;
class CkSecureStringW;
class CkSshW;
class CkBinDataW;
class CkStringBuilderW;
class CkDateTimeW;
class CkJsonObjectW;
class CkSFtpDirW;
class CkSFtpProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkSFtpW
class CK_VISIBLE_PUBLIC CkSFtpW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkSFtpW(const CkSFtpW &);
	CkSFtpW &operator=(const CkSFtpW &);

    public:
	CkSFtpW(void);
	virtual ~CkSFtpW(void);

	

	static CkSFtpW *createNew(void);
	

	CkSFtpW(bool bCallbackOwned);
	static CkSFtpW *createNew(bool bCallbackOwned);

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	CkSFtpProgressW *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkSFtpProgressW *progress);


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

	// Contains the bytes downloaded from a remote file via the AccumulateBytes method
	// call. Each call to AccumulateBytes appends to this buffer. To clear this buffer,
	// call the ClearAccumulateBuffer method.
	void get_AccumulateBuffer(CkByteData &outBytes);

	// Set to one of the following values if a call to AuthenticatePw, AuthenticatePk,
	// or AuthenticatePwPk returns a failed status.
	//     1: Transport failure. This is a failure to communicate with the server (i.e.
	//     the connection was lost, or a read or write failed or timed out).
	//     2: Invalid key for public key authentication. The key was not a valid
	//     format, not a valid key, not a private key, or not the right type of key.
	//     3: No matching authentication methods were available.
	//     4: SSH authentication protocol error - an unexpected or invalid message was
	//     received.
	//     5: The incorrect password or private key was provided.
	//     6: Already authenticated. The SSH session is already authenticated.
	//     7: Password change request: The server requires the password to be changed.
	int get_AuthFailReason(void);

	// If non-zero, limits (throttles) the download bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	int get_BandwidthThrottleDown(void);
	// If non-zero, limits (throttles) the download bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	void put_BandwidthThrottleDown(int newVal);

	// If non-zero, limits (throttles) the upload bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	int get_BandwidthThrottleUp(void);
	// If non-zero, limits (throttles) the upload bandwidth to approximately this
	// maximum number of bytes per second. The default value of this property is 0.
	void put_BandwidthThrottleUp(int newVal);

	// The client-identifier string to be used when connecting to an SSH/SFTP server.
	// Defaults to "SSH-2.0-PuTTY_Release_0.63". (This string is used to mimic PuTTY
	// because some servers are known to disconnect from clients with unknown client
	// identifiers.)
	void get_ClientIdentifier(CkString &str);
	// The client-identifier string to be used when connecting to an SSH/SFTP server.
	// Defaults to "SSH-2.0-PuTTY_Release_0.63". (This string is used to mimic PuTTY
	// because some servers are known to disconnect from clients with unknown client
	// identifiers.)
	const wchar_t *clientIdentifier(void);
	// The client-identifier string to be used when connecting to an SSH/SFTP server.
	// Defaults to "SSH-2.0-PuTTY_Release_0.63". (This string is used to mimic PuTTY
	// because some servers are known to disconnect from clients with unknown client
	// identifiers.)
	void put_ClientIdentifier(const wchar_t *newVal);

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
	const wchar_t *clientIpAddress(void);
	// The IP address to use for computers with multiple network interfaces or IP
	// addresses. For computers with a single network interface (i.e. most computers),
	// this property should not be set. For multihoming computers, the default IP
	// address is automatically used if this property is not set.
	// 
	// The IP address is a string such as in dotted notation using numbers, not domain
	// names, such as "165.164.55.124".
	// 
	void put_ClientIpAddress(const wchar_t *newVal);

	// Maximum number of milliseconds to wait when connecting to an SSH server.
	// 
	// To clarify: This property determines how long to wait for the SSH server to
	// accept the TCP connection. Once the connection is made, it is the IdleTimeoutMs
	// property that applies to receiving data and responses.
	// 
	int get_ConnectTimeoutMs(void);
	// Maximum number of milliseconds to wait when connecting to an SSH server.
	// 
	// To clarify: This property determines how long to wait for the SSH server to
	// accept the TCP connection. Once the connection is made, it is the IdleTimeoutMs
	// property that applies to receiving data and responses.
	// 
	void put_ConnectTimeoutMs(int newVal);

	// If the SSH/SFTP server sent a DISCONNECT message when closing the connection,
	// this property contains the "reason code" as specified in RFC 4253:
	//            Symbolic name                                reason code
	//            -------------                                -----------
	//       SSH_DISCONNECT_HOST_NOT_ALLOWED_TO_CONNECT             1
	//       SSH_DISCONNECT_PROTOCOL_ERROR                          2
	//       SSH_DISCONNECT_KEY_EXCHANGE_FAILED                     3
	//       SSH_DISCONNECT_RESERVED                                4
	//       SSH_DISCONNECT_MAC_ERROR                               5
	//       SSH_DISCONNECT_COMPRESSION_ERROR                       6
	//       SSH_DISCONNECT_SERVICE_NOT_AVAILABLE                   7
	//       SSH_DISCONNECT_PROTOCOL_VERSION_NOT_SUPPORTED          8
	//       SSH_DISCONNECT_HOST_KEY_NOT_VERIFIABLE                 9
	//       SSH_DISCONNECT_CONNECTION_LOST                        10
	//       SSH_DISCONNECT_BY_APPLICATION                         11
	//       SSH_DISCONNECT_TOO_MANY_CONNECTIONS                   12
	//       SSH_DISCONNECT_AUTH_CANCELLED_BY_USER                 13
	//       SSH_DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE         14
	//       SSH_DISCONNECT_ILLEGAL_USER_NAME                      15
	int get_DisconnectCode(void);

	// If the SSH/SFTP server sent a DISCONNECT message when closing the connection,
	// this property contains a descriptive string for the "reason code" as specified
	// in RFC 4253.
	void get_DisconnectReason(CkString &str);
	// If the SSH/SFTP server sent a DISCONNECT message when closing the connection,
	// this property contains a descriptive string for the "reason code" as specified
	// in RFC 4253.
	const wchar_t *disconnectReason(void);

	// Controls whether the component keeps an internal file size & attribute cache.
	// The cache affects the following methods: GetFileSize32, GetFileSize64,
	// GetFileSizeStr, GetFileCreateTime, GetFileLastAccess, GetFileLastModified,
	// GetFileOwner, GetFileGroup, and GetFilePermissions.
	// 
	// The file attribute cache exists to minimize communications with the SFTP server.
	// If the cache is enabled, then a request for any single attribute will cause all
	// of the attributes to be cached. A subsequent request for a different attribute
	// of the same file will be fulfilled from cache, eliminating the need for a
	// message to be sent to the SFTP server.
	// 
	// Note: Caching only occurs when filenames are used, not handles.
	// 
	bool get_EnableCache(void);
	// Controls whether the component keeps an internal file size & attribute cache.
	// The cache affects the following methods: GetFileSize32, GetFileSize64,
	// GetFileSizeStr, GetFileCreateTime, GetFileLastAccess, GetFileLastModified,
	// GetFileOwner, GetFileGroup, and GetFilePermissions.
	// 
	// The file attribute cache exists to minimize communications with the SFTP server.
	// If the cache is enabled, then a request for any single attribute will cause all
	// of the attributes to be cached. A subsequent request for a different attribute
	// of the same file will be fulfilled from cache, eliminating the need for a
	// message to be sent to the SFTP server.
	// 
	// Note: Caching only occurs when filenames are used, not handles.
	// 
	void put_EnableCache(bool newVal);

	// Enables or disables the use of compression w/ the SSH connection. The default
	// value is false.
	// 
	// Some older SSH servers have been found that claim to support compression, but
	// actually fail when compression is used. PuTTY does not enable compression by
	// default. If trouble is encountered where the SSH server disconnects immediately
	// after the connection is seemingly established (i.e. during authentication), then
	// check to see if disabling compression resolves the problem.
	// 
	bool get_EnableCompression(void);
	// Enables or disables the use of compression w/ the SSH connection. The default
	// value is false.
	// 
	// Some older SSH servers have been found that claim to support compression, but
	// actually fail when compression is used. PuTTY does not enable compression by
	// default. If trouble is encountered where the SSH server disconnects immediately
	// after the connection is seemingly established (i.e. during authentication), then
	// check to see if disabling compression resolves the problem.
	// 
	void put_EnableCompression(bool newVal);

	// Automatically set during the InitializeSftp method if the server sends a
	// filename-charset name-value extension. Otherwise, may be set directly to the
	// name of a charset, such as "utf-8", "iso-8859-1", "windows-1252", etc. ("ansi"
	// is also a valid choice.) Incoming filenames are interpreted as utf-8 if no
	// FilenameCharset is set. Outgoing filenames are sent using utf-8 by default.
	// Otherwise, incoming and outgoing filenames use the charset specified by this
	// property.
	void get_FilenameCharset(CkString &str);
	// Automatically set during the InitializeSftp method if the server sends a
	// filename-charset name-value extension. Otherwise, may be set directly to the
	// name of a charset, such as "utf-8", "iso-8859-1", "windows-1252", etc. ("ansi"
	// is also a valid choice.) Incoming filenames are interpreted as utf-8 if no
	// FilenameCharset is set. Outgoing filenames are sent using utf-8 by default.
	// Otherwise, incoming and outgoing filenames use the charset specified by this
	// property.
	const wchar_t *filenameCharset(void);
	// Automatically set during the InitializeSftp method if the server sends a
	// filename-charset name-value extension. Otherwise, may be set directly to the
	// name of a charset, such as "utf-8", "iso-8859-1", "windows-1252", etc. ("ansi"
	// is also a valid choice.) Incoming filenames are interpreted as utf-8 if no
	// FilenameCharset is set. Outgoing filenames are sent using utf-8 by default.
	// Otherwise, incoming and outgoing filenames use the charset specified by this
	// property.
	void put_FilenameCharset(const wchar_t *newVal);

	// Set to one of the following encryption algorithms to force that cipher to be
	// used. By default, the component will automatically choose the first cipher
	// supported by the server in the order listed here: "aes256-ctr", "aes128-ctr",
	// "aes256-cbc", "aes128-cbc", "twofish256-cbc", "twofish128-cbc", "blowfish-cbc",
	// "3des-cbc", "arcfour128", "arcfour256". (If blowfish is chosen, the encryption
	// strength is 128 bits.)
	// 
	// Important: If this is property is set and the server does NOT support then
	// encryption algorithm, then the Connect will fail.
	// 
	void get_ForceCipher(CkString &str);
	// Set to one of the following encryption algorithms to force that cipher to be
	// used. By default, the component will automatically choose the first cipher
	// supported by the server in the order listed here: "aes256-ctr", "aes128-ctr",
	// "aes256-cbc", "aes128-cbc", "twofish256-cbc", "twofish128-cbc", "blowfish-cbc",
	// "3des-cbc", "arcfour128", "arcfour256". (If blowfish is chosen, the encryption
	// strength is 128 bits.)
	// 
	// Important: If this is property is set and the server does NOT support then
	// encryption algorithm, then the Connect will fail.
	// 
	const wchar_t *forceCipher(void);
	// Set to one of the following encryption algorithms to force that cipher to be
	// used. By default, the component will automatically choose the first cipher
	// supported by the server in the order listed here: "aes256-ctr", "aes128-ctr",
	// "aes256-cbc", "aes128-cbc", "twofish256-cbc", "twofish128-cbc", "blowfish-cbc",
	// "3des-cbc", "arcfour128", "arcfour256". (If blowfish is chosen, the encryption
	// strength is 128 bits.)
	// 
	// Important: If this is property is set and the server does NOT support then
	// encryption algorithm, then the Connect will fail.
	// 
	void put_ForceCipher(const wchar_t *newVal);

	// If set to true, forces the client to choose version 3 of the SFTP protocol,
	// even if the server supports a higher version. The default value of this property
	// is true.
	bool get_ForceV3(void);
	// If set to true, forces the client to choose version 3 of the SFTP protocol,
	// even if the server supports a higher version. The default value of this property
	// is true.
	void put_ForceV3(bool newVal);

	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any SFTP operation prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	int get_HeartbeatMs(void);
	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any SFTP operation prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	void put_HeartbeatMs(int newVal);

	// Indicates the preferred host key algorithm to be used in establishing the SSH
	// secure connection. The default is "DSS". It may be changed to "RSA" if needed.
	// Chilkat recommends not changing this unless a problem warrants the change.
	void get_HostKeyAlg(CkString &str);
	// Indicates the preferred host key algorithm to be used in establishing the SSH
	// secure connection. The default is "DSS". It may be changed to "RSA" if needed.
	// Chilkat recommends not changing this unless a problem warrants the change.
	const wchar_t *hostKeyAlg(void);
	// Indicates the preferred host key algorithm to be used in establishing the SSH
	// secure connection. The default is "DSS". It may be changed to "RSA" if needed.
	// Chilkat recommends not changing this unless a problem warrants the change.
	void put_HostKeyAlg(const wchar_t *newVal);

	// Set after connecting to an SSH/SFTP server. The format of the fingerprint looks
	// like this: "ssh-rsa 1024 68:ff:d1:4e:6c:ff:d7:b0:d6:58:73:85:07:bc:2e:d5"
	void get_HostKeyFingerprint(CkString &str);
	// Set after connecting to an SSH/SFTP server. The format of the fingerprint looks
	// like this: "ssh-rsa 1024 68:ff:d1:4e:6c:ff:d7:b0:d6:58:73:85:07:bc:2e:d5"
	const wchar_t *hostKeyFingerprint(void);

	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	void get_HttpProxyAuthMethod(CkString &str);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	const wchar_t *httpProxyAuthMethod(void);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy authentication method name. Valid choices are "Basic" or "NTLM".
	void put_HttpProxyAuthMethod(const wchar_t *newVal);

	// The NTLM authentication domain (optional) if NTLM authentication is used w/ the
	// HTTP proxy.
	void get_HttpProxyDomain(CkString &str);
	// The NTLM authentication domain (optional) if NTLM authentication is used w/ the
	// HTTP proxy.
	const wchar_t *httpProxyDomain(void);
	// The NTLM authentication domain (optional) if NTLM authentication is used w/ the
	// HTTP proxy.
	void put_HttpProxyDomain(const wchar_t *newVal);

	// If an HTTP proxy is to be used, set this property to the HTTP proxy hostname or
	// IPv4 address (in dotted decimal notation).
	void get_HttpProxyHostname(CkString &str);
	// If an HTTP proxy is to be used, set this property to the HTTP proxy hostname or
	// IPv4 address (in dotted decimal notation).
	const wchar_t *httpProxyHostname(void);
	// If an HTTP proxy is to be used, set this property to the HTTP proxy hostname or
	// IPv4 address (in dotted decimal notation).
	void put_HttpProxyHostname(const wchar_t *newVal);

	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy password.
	void get_HttpProxyPassword(CkString &str);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy password.
	const wchar_t *httpProxyPassword(void);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy password.
	void put_HttpProxyPassword(const wchar_t *newVal);

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
	const wchar_t *httpProxyUsername(void);
	// If an HTTP proxy requiring authentication is to be used, set this property to
	// the HTTP proxy login name.
	void put_HttpProxyUsername(const wchar_t *newVal);

	// Causes SFTP operations to fail when progress for sending or receiving data halts
	// for more than this number of milliseconds. Setting IdleTimeoutMs = 0 allows the
	// application to wait indefinitely. The default value of this property is 30000
	// (which equals 30 seconds).
	int get_IdleTimeoutMs(void);
	// Causes SFTP operations to fail when progress for sending or receiving data halts
	// for more than this number of milliseconds. Setting IdleTimeoutMs = 0 allows the
	// application to wait indefinitely. The default value of this property is 30000
	// (which equals 30 seconds).
	void put_IdleTimeoutMs(int newVal);

	// If true, then the ReadDir method will include the "." and ".." directories in
	// its results. The default value of this property is false.
	bool get_IncludeDotDirs(void);
	// If true, then the ReadDir method will include the "." and ".." directories in
	// its results. The default value of this property is false.
	void put_IncludeDotDirs(bool newVal);

	// The InitializeSftp method call opens a channel for the SFTP session. If the
	// request to open a channel fails, this property contains a code that identifies
	// the reason for failure. The reason codes are defined in RFC 4254 and are
	// reproduced here:
	//              Symbolic name                           reason code
	//              -------------                           -----------
	//             SSH_OPEN_ADMINISTRATIVELY_PROHIBITED          1
	//             SSH_OPEN_CONNECT_FAILED                       2
	//             SSH_OPEN_UNKNOWN_CHANNEL_TYPE                 3
	//             SSH_OPEN_RESOURCE_SHORTAGE                    4
	int get_InitializeFailCode(void);

	// The InitializeSftp method call opens a channel for the SFTP session. If the
	// request to open a channel fails, this property contains a description of the
	// reason for failure. (It contains descriptive text matching the
	// InitializeFailCode property.)
	void get_InitializeFailReason(CkString &str);
	// The InitializeSftp method call opens a channel for the SFTP session. If the
	// request to open a channel fails, this property contains a description of the
	// reason for failure. (It contains descriptive text matching the
	// InitializeFailCode property.)
	const wchar_t *initializeFailReason(void);

	// Returns true if connected to the SSH server. Note: This does not mean
	// authentication has happened or InitializeSftp has already succeeded. It only
	// means that the connection has been established by calling Connect.
	// 
	// Understanding the IsConnected property: The IsConnected property is the last
	// known state of the TCP connection (either connected or disconnected). This
	// requires some explanation because most developer have incorrect assumptions
	// about TCP connections.
	//     If a TCP connection is established, and neither side is reading or writing
	//     the socket (i.e. both sides are doing nothing), then you can disconnect the
	//     network cable from the computer for any length of time, and then re-connect, and
	//     the TCP connection is not affected.
	//     A TCP connection only becomes disconnected when an attempt is made to
	//     read/write while a network problem exists. If no attempts to read/write occur, a
	//     network problem may arise and then become resolved without affecting the TCP
	//     connection.
	//     If the peer chooses to close its side of the TCP connection, your
	//     application won't magically know about it until you try to do something with the
	//     TCP socket (such as read or write).
	//     A Chilkat API property (as opposed to a method) CANNOT and should not do
	//     something that would cause an application to timeout, hang, etc. Therefore, it
	//     is not appropriate for the IsConnected property to attempt any kind of socket
	//     operation (read/write/peek) on the socket. It simply returns the last known
	//     state of the connection. It may very well be that your network cable is
	//     unplugged and IsConnected returns true because technically, if neither peer is
	//     trying to read/write, the network cable could be plugged back in without
	//     affecting the connection. IsConnected could also return true if the peer has
	//     already closed its side of the connection, because the state of the connection
	//     is only updated after trying to read/write/peek.
	//     To truly know the current connected state (as opposed to the last known
	//     connection state), your application should attempt a network operation that is
	//     appropriate to the protocol. For SFTP, an application could call SendIgnore, and
	//     then check IsConnected.
	// 
	bool get_IsConnected(void);

	// Controls whether communications to/from the SFTP server are saved to the
	// SessionLog property. The default value is false. If this property is set to
	// true, the contents of the SessionLog property will continuously grow as SFTP
	// activity transpires. The purpose of the KeepSessionLog / SessionLog properties
	// is to help in debugging any future problems that may arise.
	bool get_KeepSessionLog(void);
	// Controls whether communications to/from the SFTP server are saved to the
	// SessionLog property. The default value is false. If this property is set to
	// true, the contents of the SessionLog property will continuously grow as SFTP
	// activity transpires. The purpose of the KeepSessionLog / SessionLog properties
	// is to help in debugging any future problems that may arise.
	void put_KeepSessionLog(bool newVal);

	// Contains the numeric last status code received in an SSH_FXP_STATUS message
	// received from the SFTP server. Status codes are defined in the SFTP protocol
	// specification as follows:
	// 
	// (Seehttps://tools.ietf.org/id/draft-ietf-secsh-filexfer-13.txt)
	// SSH_FX_OK                            0
	// SSH_FX_EOF                           1
	// SSH_FX_NO_SUCH_FILE                  2
	// SSH_FX_PERMISSION_DENIED             3
	// SSH_FX_FAILURE                       4
	// SSH_FX_BAD_MESSAGE                   5
	// SSH_FX_NO_CONNECTION                 6
	// SSH_FX_CONNECTION_LOST               7
	// SSH_FX_OP_UNSUPPORTED                8
	// SSH_FX_INVALID_HANDLE                9
	// SSH_FX_NO_SUCH_PATH                  10
	// SSH_FX_FILE_ALREADY_EXISTS           11
	// SSH_FX_WRITE_PROTECT                 12
	// SSH_FX_NO_MEDIA                      13
	// SSH_FX_NO_SPACE_ON_FILESYSTEM        14
	// SSH_FX_QUOTA_EXCEEDED                15
	// SSH_FX_UNKNOWN_PRINCIPAL             16
	// SSH_FX_LOCK_CONFLICT                 17
	// SSH_FX_DIR_NOT_EMPTY                 18
	// SSH_FX_NOT_A_DIRECTORY               19
	// SSH_FX_INVALID_FILENAME              20
	// SSH_FX_LINK_LOOP                     21
	// SSH_FX_CANNOT_DELETE                 22
	// SSH_FX_INVALID_PARAMETER             23
	// SSH_FX_FILE_IS_A_DIRECTORY           24
	// SSH_FX_BYTE_RANGE_LOCK_CONFLICT      25
	// SSH_FX_BYTE_RANGE_LOCK_REFUSED       26
	// SSH_FX_DELETE_PENDING                27
	// SSH_FX_FILE_CORRUPT                  28
	// SSH_FX_OWNER_INVALID                 29
	// SSH_FX_GROUP_INVALID                 30
	// SSH_FX_NO_MATCHING_BYTE_RANGE_LOCK       31
	// 
	int get_LastStatusCode(void);

	// Contains the message text received in the last SSH_FXP_STATUS message received
	// from the SFTP server. The actual text of the message can vary depending on the
	// SFTP server implementation. The integer status code is available in the
	// LastStatusCode property.
	void get_LastStatusMessage(CkString &str);
	// Contains the message text received in the last SSH_FXP_STATUS message received
	// from the SFTP server. The actual text of the message can vary depending on the
	// SFTP server implementation. The integer status code is available in the
	// LastStatusCode property.
	const wchar_t *lastStatusMessage(void);

	// The maximum packet length to be used in the underlying SSH transport protocol.
	// The default value is 32768. (This should generally be left unchanged.)
	int get_MaxPacketSize(void);
	// The maximum packet length to be used in the underlying SSH transport protocol.
	// The default value is 32768. (This should generally be left unchanged.)
	void put_MaxPacketSize(int newVal);

	// Set by the AuthenticatePw and AuthenticatePwPk methods. If the authenticate
	// method returns a failed status, and this property is set to true, then it
	// indicates the server requested a password change. In this case, re-call the
	// authenticate method, but provide both the old and new passwords in the following
	// format, where vertical bar characters encapsulate the old and new passwords:
	// 
	//     |oldPassword|newPassword|
	// 
	bool get_PasswordChangeRequested(void);

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

	// If true, then the file last-modified and create date/time will be preserved
	// for downloaded and uploaded files. The default value is false.
	// 
	// Note: Prior to version 9.5.0.40, this property only applied to downloads.
	// Starting in v9.5.0.40, it also applies to the UploadFileByName method.
	//     It does not apply to uploads or downloads where the remote file is opened to
	//     obtain a handle, the data is uploaded/downloaded, and then the handle is closed.
	//     The last-mod date/times are always preserved ini the SyncTreeDownload and
	//     SyncTreeUpload methods.
	// 
	bool get_PreserveDate(void);
	// If true, then the file last-modified and create date/time will be preserved
	// for downloaded and uploaded files. The default value is false.
	// 
	// Note: Prior to version 9.5.0.40, this property only applied to downloads.
	// Starting in v9.5.0.40, it also applies to the UploadFileByName method.
	//     It does not apply to uploads or downloads where the remote file is opened to
	//     obtain a handle, the data is uploaded/downloaded, and then the handle is closed.
	//     The last-mod date/times are always preserved ini the SyncTreeDownload and
	//     SyncTreeUpload methods.
	// 
	void put_PreserveDate(bool newVal);

	// The negotiated SFTP protocol version, which should be a value between 3 and 6
	// inclusive. When the InitializeSftp method is called, the Chilkat SFTP client
	// sends it's highest supported protocol version to the server, and the server
	// sends it's SFTP protocol version in response. The negotiated protocol (i.e. the
	// protocol version used for the session) is the lower of the two numbers. If the
	// SFTP server's protocol version is lower than 6, some file date/attributes are
	// not supported because they are not supported by the earlier protocol version.
	// These exceptions are noted throughout the reference documentation.
	int get_ProtocolVersion(void);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the ReadDir method will only return
	// files that match any one of these patterns.
	void get_ReadDirMustMatch(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the ReadDir method will only return
	// files that match any one of these patterns.
	const wchar_t *readDirMustMatch(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the ReadDir method will only return
	// files that match any one of these patterns.
	void put_ReadDirMustMatch(const wchar_t *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the ReadDir method will only return
	// files that do not match any of these patterns.
	void get_ReadDirMustNotMatch(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the ReadDir method will only return
	// files that do not match any of these patterns.
	const wchar_t *readDirMustNotMatch(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the ReadDir method will only return
	// files that do not match any of these patterns.
	void put_ReadDirMustNotMatch(const wchar_t *newVal);

	// The server-identifier string received from the server during connection
	// establishment. For example, a typical value would be similar to
	// "SSH-2.0-OpenSSH_7.2p2 Ubuntu-4ubuntu2.2".
	void get_ServerIdentifier(CkString &str);
	// The server-identifier string received from the server during connection
	// establishment. For example, a typical value would be similar to
	// "SSH-2.0-OpenSSH_7.2p2 Ubuntu-4ubuntu2.2".
	const wchar_t *serverIdentifier(void);

	// Contains a log of the messages sent to/from the SFTP server. To enable session
	// logging, set the KeepSessionLog property = true. Note: This property is not a
	// filename -- it is a string property that contains the session log data.
	void get_SessionLog(CkString &str);
	// Contains a log of the messages sent to/from the SFTP server. To enable session
	// logging, set the KeepSessionLog property = true. Note: This property is not a
	// filename -- it is a string property that contains the session log data.
	const wchar_t *sessionLog(void);

	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	void get_SocksHostname(CkString &str);
	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	const wchar_t *socksHostname(void);
	// The SOCKS4/SOCKS5 hostname or IPv4 address (in dotted decimal notation). This
	// property is only used if the SocksVersion property is set to 4 or 5).
	void put_SocksHostname(const wchar_t *newVal);

	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	void get_SocksPassword(CkString &str);
	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	const wchar_t *socksPassword(void);
	// The SOCKS5 password (if required). The SOCKS4 protocol does not include the use
	// of a password, so this does not apply to SOCKS4.
	void put_SocksPassword(const wchar_t *newVal);

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
	const wchar_t *socksUsername(void);
	// The SOCKS4/SOCKS5 proxy username. This property is only used if the SocksVersion
	// property is set to 4 or 5).
	void put_SocksUsername(const wchar_t *newVal);

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

	// A property that can contain a list of comma-separated keywords to control
	// certain aspects of an upload or download synchronization (for the SyncTreeUpload
	// and SyncTreeDownload methods). At this time there is only one possible
	// directive, but others may be added in the future.
	// 
	// Set this property to "UploadIgnoreLocalOpenFailures" to skip local files that
	// cannot be opened. A common reason for this would be if another process on the
	// system has the file open for exclusive access.
	// 
	void get_SyncDirectives(CkString &str);
	// A property that can contain a list of comma-separated keywords to control
	// certain aspects of an upload or download synchronization (for the SyncTreeUpload
	// and SyncTreeDownload methods). At this time there is only one possible
	// directive, but others may be added in the future.
	// 
	// Set this property to "UploadIgnoreLocalOpenFailures" to skip local files that
	// cannot be opened. A common reason for this would be if another process on the
	// system has the file open for exclusive access.
	// 
	const wchar_t *syncDirectives(void);
	// A property that can contain a list of comma-separated keywords to control
	// certain aspects of an upload or download synchronization (for the SyncTreeUpload
	// and SyncTreeDownload methods). At this time there is only one possible
	// directive, but others may be added in the future.
	// 
	// Set this property to "UploadIgnoreLocalOpenFailures" to skip local files that
	// cannot be opened. A common reason for this would be if another process on the
	// system has the file open for exclusive access.
	// 
	void put_SyncDirectives(const wchar_t *newVal);

	// The paths of the files uploaded or downloaded in the last call to SyncUploadTree
	// or SyncDownloadTree. The paths are listed one per line. In both cases (for
	// upload and download) each line contains the paths relative to the root synced
	// directory.
	// 
	// Note: For SyncTreeDownload, some of entires can be the paths of local
	// directories that were created. Starting in v9.5.0.77, local directory paths will
	// be terminated with a "/" char (to disinguish a directory from an actual file).
	// 
	void get_SyncedFiles(CkString &str);
	// The paths of the files uploaded or downloaded in the last call to SyncUploadTree
	// or SyncDownloadTree. The paths are listed one per line. In both cases (for
	// upload and download) each line contains the paths relative to the root synced
	// directory.
	// 
	// Note: For SyncTreeDownload, some of entires can be the paths of local
	// directories that were created. Starting in v9.5.0.77, local directory paths will
	// be terminated with a "/" char (to disinguish a directory from an actual file).
	// 
	const wchar_t *syncedFiles(void);
	// The paths of the files uploaded or downloaded in the last call to SyncUploadTree
	// or SyncDownloadTree. The paths are listed one per line. In both cases (for
	// upload and download) each line contains the paths relative to the root synced
	// directory.
	// 
	// Note: For SyncTreeDownload, some of entires can be the paths of local
	// directories that were created. Starting in v9.5.0.77, local directory paths will
	// be terminated with a "/" char (to disinguish a directory from an actual file).
	// 
	void put_SyncedFiles(const wchar_t *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only transfer files that match any one of these patterns. This
	// property only applies to files. It does not apply to sub-directory names when
	// recursively traversing a directory tree.
	void get_SyncMustMatch(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only transfer files that match any one of these patterns. This
	// property only applies to files. It does not apply to sub-directory names when
	// recursively traversing a directory tree.
	const wchar_t *syncMustMatch(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only transfer files that match any one of these patterns. This
	// property only applies to files. It does not apply to sub-directory names when
	// recursively traversing a directory tree.
	void put_SyncMustMatch(const wchar_t *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only enter directories that match any one of these patterns.
	void get_SyncMustMatchDir(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only enter directories that match any one of these patterns.
	const wchar_t *syncMustMatchDir(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will only enter directories that match any one of these patterns.
	void put_SyncMustMatchDir(const wchar_t *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not transfer files that match any one of these patterns. This
	// property only applies to files. It does not apply to sub-directory names when
	// recursively traversing a directory tree.
	void get_SyncMustNotMatch(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not transfer files that match any one of these patterns. This
	// property only applies to files. It does not apply to sub-directory names when
	// recursively traversing a directory tree.
	const wchar_t *syncMustNotMatch(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "*.xml; *.txt; *.csv". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not transfer files that match any one of these patterns. This
	// property only applies to files. It does not apply to sub-directory names when
	// recursively traversing a directory tree.
	void put_SyncMustNotMatch(const wchar_t *newVal);

	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not enter directories that match any one of these patterns.
	void get_SyncMustNotMatchDir(CkString &str);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not enter directories that match any one of these patterns.
	const wchar_t *syncMustNotMatchDir(void);
	// Can contain a wildcarded list of file patterns separated by semicolons. For
	// example, "xml; txt; data_*". If set, the SyncTreeUpload and SyncTreeDownload
	// methods will not enter directories that match any one of these patterns.
	void put_SyncMustNotMatchDir(const wchar_t *newVal);

	// This property controls the use of the internal TCP_NODELAY socket option (which
	// disables the Nagle algorithm). The default value of this property is false.
	// Setting this value to true disables the delay of sending successive small
	// packets on the network.
	bool get_TcpNoDelay(void);
	// This property controls the use of the internal TCP_NODELAY socket option (which
	// disables the Nagle algorithm). The default value of this property is false.
	// Setting this value to true disables the delay of sending successive small
	// packets on the network.
	void put_TcpNoDelay(bool newVal);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "KEX_DH_GEX_REQUEST_OLD" - Introduced in v9.5.0.73. Force the old Group
	//     Exchange message to be used. This would be used for very old SSH server
	//     implementations that do not use the RFC standard for
	//     diffie-hellman-group-exchange.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "NoSyncDownloadEmptyFiles" - Introduced in v9.5.0.80. Prevents empty files
	//     from being downloaded in SyncTreeDownload.
	// 
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "KEX_DH_GEX_REQUEST_OLD" - Introduced in v9.5.0.73. Force the old Group
	//     Exchange message to be used. This would be used for very old SSH server
	//     implementations that do not use the RFC standard for
	//     diffie-hellman-group-exchange.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "NoSyncDownloadEmptyFiles" - Introduced in v9.5.0.80. Prevents empty files
	//     from being downloaded in SyncTreeDownload.
	// 
	const wchar_t *uncommonOptions(void);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string, and should typically remain empty.
	// 
	// Can be set to a list of the following comma separated keywords:
	//     "KEX_DH_GEX_REQUEST_OLD" - Introduced in v9.5.0.73. Force the old Group
	//     Exchange message to be used. This would be used for very old SSH server
	//     implementations that do not use the RFC standard for
	//     diffie-hellman-group-exchange.
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	//     "NoSyncDownloadEmptyFiles" - Introduced in v9.5.0.80. Prevents empty files
	//     from being downloaded in SyncTreeDownload.
	// 
	void put_UncommonOptions(const wchar_t *newVal);

	// The chunk size to use when uploading files via the UploadFile or
	// UploadFileByName methods. The default value is 32000. If an upload fails because
	// "WSAECONNABORTED An established connection was aborted by the software in your
	// host machine.", then try setting this property to a smaller value, such as 4096.
	// A smaller value will result in slower transfer rates, but may help avoid this
	// problem.
	int get_UploadChunkSize(void);
	// The chunk size to use when uploading files via the UploadFile or
	// UploadFileByName methods. The default value is 32000. If an upload fails because
	// "WSAECONNABORTED An established connection was aborted by the software in your
	// host machine.", then try setting this property to a smaller value, such as 4096.
	// A smaller value will result in slower transfer rates, but may help avoid this
	// problem.
	void put_UploadChunkSize(int newVal);

	// If a user authentication banner message is received during authentication, it
	// will be made available here. An application can check to see if this property
	// contains a banner string after calling StartKeyboardAuth. It is only possible
	// for an application to display this message if it is doing keyboard-interactive
	// authentication via the StartKeyboardAuth and ContinueKeyboardAuth methods.
	void get_UserAuthBanner(CkString &str);
	// If a user authentication banner message is received during authentication, it
	// will be made available here. An application can check to see if this property
	// contains a banner string after calling StartKeyboardAuth. It is only possible
	// for an application to display this message if it is doing keyboard-interactive
	// authentication via the StartKeyboardAuth and ContinueKeyboardAuth methods.
	const wchar_t *userAuthBanner(void);
	// If a user authentication banner message is received during authentication, it
	// will be made available here. An application can check to see if this property
	// contains a banner string after calling StartKeyboardAuth. It is only possible
	// for an application to display this message if it is doing keyboard-interactive
	// authentication via the StartKeyboardAuth and ContinueKeyboardAuth methods.
	void put_UserAuthBanner(const wchar_t *newVal);

	// If true, then date/times are returned as UTC times. If false (the default)
	// then date/times are returned as local times.
	bool get_UtcMode(void);
	// If true, then date/times are returned as UTC times. If false (the default)
	// then date/times are returned as local times.
	void put_UtcMode(bool newVal);

	// The current transfer byte count for an ongoing upload or download. Programs
	// doing asynchronous uploads or downloads can read this property in real time
	// during the upload. For SyncTreeUpload and SyncTreeDownload operations, this is
	// the real-time cumulative number of bytes for all files uploaded or downloaded.
	unsigned long get_XferByteCount(void);

	// The current transfer byte count for an ongoing upload or download. Programs
	// doing asynchronous uploads or downloads can read this property in real time
	// during the upload. For SyncTreeUpload and SyncTreeDownload operations, this is
	// the real-time cumulative number of bytes for all files uploaded or downloaded.
	__int64 get_XferByteCount64(void);



	// ----------------------
	// Methods
	// ----------------------
	// Downloads bytes from an open file and appends them to the AccumulateBuffer. The
	// handle is a file handle returned by the OpenFile method. The maxBytes is the maximum
	// number of bytes to read. If the end-of-file is reached prior to reading the
	// number of requested bytes, then fewer bytes may be returned.
	// 
	// Returns the number of bytes downloaded and appended to AccumulateBuffer. Returns
	// -1 on error.
	// 
	int AccumulateBytes(const wchar_t *handle, int maxBytes);

	// Creates an asynchronous task to call the AccumulateBytes method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *AccumulateBytesAsync(const wchar_t *handle, int maxBytes);

	// Convenience method for 64-bit addition. Allows for two 64-bit integers to be
	// passed as decimal strings and returns the sum in a decimal sting.
	bool Add64(const wchar_t *n1, const wchar_t *n2, CkString &outStr);
	// Convenience method for 64-bit addition. Allows for two 64-bit integers to be
	// passed as decimal strings and returns the sum in a decimal sting.
	const wchar_t *add64(const wchar_t *n1, const wchar_t *n2);

	// Authenticates with the SSH server using public-key authentication. The
	// corresponding public key must have been installed on the SSH server for the
	// username. Authentication will succeed if the matching privateKey is provided.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	bool AuthenticatePk(const wchar_t *username, CkSshKeyW &privateKey);

	// Creates an asynchronous task to call the AuthenticatePk method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *AuthenticatePkAsync(const wchar_t *username, CkSshKeyW &privateKey);

	// Authenticates with the SSH server using a login and password.
	// 
	// An SFTP session always begins by first calling Connect to connect to the SSH
	// server, then calling either AuthenticatePw or AuthenticatePk to login, and
	// finally calling InitializeSftp.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	// Note: To learn about how to handle password change requests, see the
	// PasswordChangeRequested property (above).
	// 
	bool AuthenticatePw(const wchar_t *login, const wchar_t *password);

	// Creates an asynchronous task to call the AuthenticatePw method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *AuthenticatePwAsync(const wchar_t *login, const wchar_t *password);

	// Authentication for SSH servers that require both a password and private key.
	// (Most SSH servers are configured to require one or the other, but not both.)
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	bool AuthenticatePwPk(const wchar_t *username, const wchar_t *password, CkSshKeyW &privateKey);

	// Creates an asynchronous task to call the AuthenticatePwPk method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *AuthenticatePwPkAsync(const wchar_t *username, const wchar_t *password, CkSshKeyW &privateKey);

	// The same as AuthenticatePw, but the login and password are passed in secure
	// string objects.
	bool AuthenticateSecPw(CkSecureStringW &login, CkSecureStringW &password);

	// Creates an asynchronous task to call the AuthenticateSecPw method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *AuthenticateSecPwAsync(CkSecureStringW &login, CkSecureStringW &password);

	// The same as AuthenticatePwPk, but the login and password are passed in secure
	// string objects.
	bool AuthenticateSecPwPk(CkSecureStringW &username, CkSecureStringW &password, CkSshKeyW &privateKey);

	// Creates an asynchronous task to call the AuthenticateSecPwPk method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *AuthenticateSecPwPkAsync(CkSecureStringW &username, CkSecureStringW &password, CkSshKeyW &privateKey);

	// Clears the contents of the AccumulateBuffer property.
	void ClearAccumulateBuffer(void);

	// Clears the internal file attribute cache. (Please refer to the EnableCache
	// property for more information about the file attribute cache.)
	void ClearCache(void);

	// Clears the contents of the SessionLog property.
	void ClearSessionLog(void);

	// Closes a file on the SSH/SFTP server. handle is a file handle returned from a
	// previous call to OpenFile or OpenDir.
	bool CloseHandle(const wchar_t *handle);

	// Creates an asynchronous task to call the CloseHandle method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CloseHandleAsync(const wchar_t *handle);

	// Connects to an SSH/SFTP server. The domainName may be a domain name or an IP address
	// (example: 192.168.1.10). Both IPv4 and IPv6 addresses are supported. The port is
	// typically 22, which is the standard port for SSH servers.
	// 
	// An SFTP session always begins by first calling Connect to connect to the SSH
	// server, then calling either AuthenticatePw or AuthenticatePk to login, and
	// finally calling InitializeSftp.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	// Important: All TCP-based Internet communications, regardless of the protocol
	// (such as HTTP, FTP, SSH, IMAP, POP3, SMTP, etc.), and regardless of SSL/TLS,
	// begin with establishing a TCP connection to a remote host:port. External
	// security-related infrastructure such as software firewalls (Windows Firewall),
	// hardware firewalls, anti-virus, at either source or destination (or both) can
	// block the connection. If the connection fails, make sure to check all potential
	// external causes of blockage.
	// 
	bool Connect(const wchar_t *domainName, int port);

	// Creates an asynchronous task to call the Connect method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ConnectAsync(const wchar_t *domainName, int port);

	// Connects to an SSH/SFTP server through an existing SSH connection. The sshConn is
	// an existing connected and authenticated SSH object. The connection to hostname:port
	// is made through the existing SSH connection via port-forwarding. If successful,
	// the connection is as follows: application => ServerSSH1 => ServerSSH2. (where
	// ServerSSH1 is the sshConn and ServerSSH2 is the SSH server at hostname:port) Once
	// connected in this way, all communications are routed through ServerSSH1 to
	// ServerSSH2. This includes authentication -- which means the application must
	// still call one of the Authenticate* methods to authenticate with ServerSSH2.
	bool ConnectThroughSsh(CkSshW &sshConn, const wchar_t *hostname, int port);

	// Creates an asynchronous task to call the ConnectThroughSsh method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ConnectThroughSshAsync(CkSshW &sshConn, const wchar_t *hostname, int port);

	// Continues keyboard-interactive authentication with the SSH server. The response is
	// typically the password. If multiple responses are required (because there were
	// multiple prompts in the infoRequest XML returned by StartKeyboardAuth), then the
	// response should be formatted as XML (as shown below) otherwise the response simply
	// contains the single response string.
	// _LT_response_GT_
	//     _LT_response1_GT_response to first prompt_LT_/response1_GT_
	//     _LT_response2_GT_response to second prompt_LT_/response2_GT_
	//     ...
	//     _LT_responseN_GT_response to Nth prompt_LT_/responseN_GT_
	// _LT_/response_GT_
	// 
	// If the interactive authentication completed with success or failure, the XML
	// response will be:
	// _LT_success_GT_success_message_LT_/success_GT_
	// 
	// or
	// 
	// _LT_error_GT_error_message_LT_/error_GT_
	// If additional steps are required to complete the interactive authentication,
	// then an XML string that provides the name, instruction, and prompts is returned.
	// The XML has the following format:
	//  	_LT_infoRequest numPrompts="N"_GT_
	// 	    _LT_name_GT_name_string_LT_/name_GT_
	// 	    _LT_instruction_GT_instruction_string_LT_/instruction_GT_
	// 	    _LT_prompt1 echo="1_or_0"_GT_prompt_string_LT_/prompt1_GT_
	// 	    ...
	// 	    _LT_promptN echo="1_or_0"_GT_prompt_string_LT_/promptN_GT_
	// 	_LT_/infoRequest_GT_
	// 
	bool ContinueKeyboardAuth(const wchar_t *response, CkString &outStr);
	// Continues keyboard-interactive authentication with the SSH server. The response is
	// typically the password. If multiple responses are required (because there were
	// multiple prompts in the infoRequest XML returned by StartKeyboardAuth), then the
	// response should be formatted as XML (as shown below) otherwise the response simply
	// contains the single response string.
	// _LT_response_GT_
	//     _LT_response1_GT_response to first prompt_LT_/response1_GT_
	//     _LT_response2_GT_response to second prompt_LT_/response2_GT_
	//     ...
	//     _LT_responseN_GT_response to Nth prompt_LT_/responseN_GT_
	// _LT_/response_GT_
	// 
	// If the interactive authentication completed with success or failure, the XML
	// response will be:
	// _LT_success_GT_success_message_LT_/success_GT_
	// 
	// or
	// 
	// _LT_error_GT_error_message_LT_/error_GT_
	// If additional steps are required to complete the interactive authentication,
	// then an XML string that provides the name, instruction, and prompts is returned.
	// The XML has the following format:
	//  	_LT_infoRequest numPrompts="N"_GT_
	// 	    _LT_name_GT_name_string_LT_/name_GT_
	// 	    _LT_instruction_GT_instruction_string_LT_/instruction_GT_
	// 	    _LT_prompt1 echo="1_or_0"_GT_prompt_string_LT_/prompt1_GT_
	// 	    ...
	// 	    _LT_promptN echo="1_or_0"_GT_prompt_string_LT_/promptN_GT_
	// 	_LT_/infoRequest_GT_
	// 
	const wchar_t *continueKeyboardAuth(const wchar_t *response);

	// Creates an asynchronous task to call the ContinueKeyboardAuth method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ContinueKeyboardAuthAsync(const wchar_t *response);

	// Sets the date/time and other attributes of a remote file to be equal to that of
	// a local file.
	// 
	// The attributes copied depend on the SFTP version of the server:
	// SFTP v3 (and below)
	//     Last-Modified Date/Time
	//     Last-Access Date/Time
	// 
	// SFTP v4, v5
	//     Last-Modified Date/Time
	//     Last-Access Date/Time
	//     Create Date/Time
	// 
	// SFTP v6 (and above)
	//     Last-Modified Date/Time
	//     Last-Access Date/Time
	//     Create Date/Time
	//     Read-Only Flag
	//     Hidden Flag
	//     Archive Flag
	//     Compressed Flag
	//     Encrypted Flag
	// 
	// Notes:
	// (1) The Last-Access date/time may or may not be set. Chilkat has found that the
	// Last-Access time is set to the current date/time, which is probably a result of
	// the operating system setting it based on when the SFTP server is touching the
	// file.
	// (2) At the time of this writing, it is unknown whether the compressed/encryption
	// settings for a local file are transferred to the remote file. For example, does
	// the remote file become compressed and/or encrypted just by setting the flags? It
	// may depend on the SFTP server and/or the remote filesystem.
	// (3) Dates/times are sent in GMT. SFTP servers should convert GMT times to local
	// time zones.
	// 
	bool CopyFileAttr(const wchar_t *localFilename, const wchar_t *remoteFilename, bool isHandle);

	// Creates an asynchronous task to call the CopyFileAttr method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CopyFileAttrAsync(const wchar_t *localFilename, const wchar_t *remoteFilename, bool isHandle);

	// Creates a directory on the SFTP server.
	bool CreateDir(const wchar_t *path);

	// Creates an asynchronous task to call the CreateDir method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *CreateDirAsync(const wchar_t *path);

	// Disconnects from the SSH server.
	void Disconnect(void);

	// Downloads the contents of a remote file to a BinData object. (Appends to the
	// BinData.)
	bool DownloadBd(const wchar_t *remoteFilePath, CkBinDataW &binData);

	// Creates an asynchronous task to call the DownloadBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *DownloadBdAsync(const wchar_t *remoteFilePath, CkBinDataW &binData);

	// Downloads a file from the SSH server to the local filesystem. There are no
	// limitations on file size and the data is streamed from SSH server to the local
	// file. handle is a file handle returned by a previous call to OpenFile.
	bool DownloadFile(const wchar_t *handle, const wchar_t *toFilename);

	// Creates an asynchronous task to call the DownloadFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *DownloadFileAsync(const wchar_t *handle, const wchar_t *toFilename);

	// Simplified method for downloading files.
	// 
	// The last-modified date/time is only preserved when the PreserveDate property is
	// set to true. (The default value of PreserveDate is false.)
	// 
	bool DownloadFileByName(const wchar_t *remoteFilePath, const wchar_t *localFilePath);

	// Creates an asynchronous task to call the DownloadFileByName method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *DownloadFileByNameAsync(const wchar_t *remoteFilePath, const wchar_t *localFilePath);

	// Downloads the contents of a remote file to a StringBuilder object. (Appends to
	// the StringBuilder.)
	bool DownloadSb(const wchar_t *remoteFilePath, const wchar_t *charset, CkStringBuilderW &sb);

	// Creates an asynchronous task to call the DownloadSb method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *DownloadSbAsync(const wchar_t *remoteFilePath, const wchar_t *charset, CkStringBuilderW &sb);

	// Returns true if the last read operation for a handle reached the end of file.
	// Otherwise returns false. If an invalid handle is passed, a value of true is
	// returned.
	bool Eof(const wchar_t *handle);

	// Returns a value to indicate whether the remote file exists or not. remotePath is the
	// path of the remote file. If followLinks is true, then symbolic links will be followed
	// on the server.
	// 
	// This method returns one of the following possible values:
	//     -1: Unable to check. Examine the LastErrorText to determine the reason for
	//     failure.
	//     0: File does not exist.
	//     1: The regular file exists.
	//     2: It exists, but it is a directory.
	//     3: It exists, but it is a symlink (only possible if followLinks is false)
	//     4: It exists, but it is a special filesystem entry type.
	//     5: It exists, but it is an unkown filesystem entry type.
	//     6: It exists, but it is an socket filesystem entry type.
	//     7: It exists, but it is an char device entry type.
	//     8: It exists, but it is an block device entry type.
	//     9: It exists, but it is an FIFO entry type.
	// 
	// Note: The values greater than zero correspond to the possible values as
	// specified in the SFTP protocol specification. A given value may not make sense
	// on all operating systems.
	// 
	int FileExists(const wchar_t *remotePath, bool followLinks);

	// Creates an asynchronous task to call the FileExists method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *FileExistsAsync(const wchar_t *remotePath, bool followLinks);

	// Causes the SFTP server to do an fsync on the open file. Specifically, this is
	// directing the SFTP server to call fsync (https://linux.die.net/man/2/fsync) on
	// the open file.
	// 
	// This method uses the fsync@openssh.com and only works for servers supporting the
	// fsync@openssh.com extension.
	// 
	bool Fsync(const wchar_t *handle);

	// Creates an asynchronous task to call the Fsync method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *FsyncAsync(const wchar_t *handle);

	// Returns the create date/time for a file. pathOrHandle may be a remote filepath or an
	// open handle string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle must
	// be set to true, otherwise it should be false. If bFollowLinks is true, then
	// symbolic links will be followed on the server.
	// 
	// Note: Servers running the SFTP v3 protocol or lower do not have the ability to
	// return a file's creation date/time.
	// 
	// The caller is responsible for deleting the object returned by this method.
	CkDateTimeW *GetFileCreateDt(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFileCreateDt method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFileCreateDtAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns the create date/time for a file. pathOrHandle may be a remote filepath or an
	// open handle string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle must
	// be set to true, otherwise it should be false. If bFollowLinks is true, then
	// symbolic links will be followed on the server.
	// 
	// Note: Servers running the SFTP v3 protocol or lower do not have the ability to
	// return a file's creation date/time.
	// 
	bool GetFileCreateTime(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, SYSTEMTIME &outSysTime);

	// The same as GetFileCreateTime, except the date/time is returned as an RFC822
	// formatted string.
	bool GetFileCreateTimeStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, CkString &outStr);
	// The same as GetFileCreateTime, except the date/time is returned as an RFC822
	// formatted string.
	const wchar_t *getFileCreateTimeStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);
	// The same as GetFileCreateTime, except the date/time is returned as an RFC822
	// formatted string.
	const wchar_t *fileCreateTimeStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFileCreateTimeStr method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFileCreateTimeStrAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns the group of a file. pathOrHandle may be a remote filepath or an open handle
	// string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle must be set to
	// true, otherwise it should be false. If bFollowLinks is true, then symbolic links
	// will be followed on the server.
	// 
	// Note: Servers running the SFTP v3 protocol or lower do not have the ability to
	// return a file's group name. Instead, the decimal GID of the file is returned.
	// 
	bool GetFileGroup(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, CkString &outStr);
	// Returns the group of a file. pathOrHandle may be a remote filepath or an open handle
	// string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle must be set to
	// true, otherwise it should be false. If bFollowLinks is true, then symbolic links
	// will be followed on the server.
	// 
	// Note: Servers running the SFTP v3 protocol or lower do not have the ability to
	// return a file's group name. Instead, the decimal GID of the file is returned.
	// 
	const wchar_t *getFileGroup(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);
	// Returns the group of a file. pathOrHandle may be a remote filepath or an open handle
	// string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle must be set to
	// true, otherwise it should be false. If bFollowLinks is true, then symbolic links
	// will be followed on the server.
	// 
	// Note: Servers running the SFTP v3 protocol or lower do not have the ability to
	// return a file's group name. Instead, the decimal GID of the file is returned.
	// 
	const wchar_t *fileGroup(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFileGroup method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFileGroupAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns the last-access date/time for a file. pathOrHandle may be a remote filepath or
	// an open handle string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle
	// must be set to true, otherwise it should be false. If bFollowLinks is true, then
	// symbolic links will be followed on the server.
	bool GetFileLastAccess(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, SYSTEMTIME &outSysTime);

	// Returns the last-access date/time for a file. pathOrHandle may be a remote filepath or
	// an open handle string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle
	// must be set to true, otherwise it should be false. If bFollowLinks is true, then
	// symbolic links will be followed on the server.
	// The caller is responsible for deleting the object returned by this method.
	CkDateTimeW *GetFileLastAccessDt(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFileLastAccessDt method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFileLastAccessDtAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// The same as GetFileLastAccess, except the date/time is returned as an RFC822
	// formatted string.
	bool GetFileLastAccessStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, CkString &outStr);
	// The same as GetFileLastAccess, except the date/time is returned as an RFC822
	// formatted string.
	const wchar_t *getFileLastAccessStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);
	// The same as GetFileLastAccess, except the date/time is returned as an RFC822
	// formatted string.
	const wchar_t *fileLastAccessStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFileLastAccessStr method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFileLastAccessStrAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns the last-modified date/time for a file. pathOrHandle may be a remote filepath or
	// an open handle string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle
	// must be set to true, otherwise it should be false. If bFollowLinks is true, then
	// symbolic links will be followed on the server.
	bool GetFileLastModified(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, SYSTEMTIME &outSysTime);

	// Returns the last-modified date/time for a file. pathOrHandle may be a remote filepath or
	// an open handle string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle
	// must be set to true, otherwise it should be false. If bFollowLinks is true, then
	// symbolic links will be followed on the server.
	// The caller is responsible for deleting the object returned by this method.
	CkDateTimeW *GetFileLastModifiedDt(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFileLastModifiedDt method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFileLastModifiedDtAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// The same as GetFileLastModified, except the date/time is returned as an RFC822
	// formatted string.
	bool GetFileLastModifiedStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, CkString &outStr);
	// The same as GetFileLastModified, except the date/time is returned as an RFC822
	// formatted string.
	const wchar_t *getFileLastModifiedStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);
	// The same as GetFileLastModified, except the date/time is returned as an RFC822
	// formatted string.
	const wchar_t *fileLastModifiedStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFileLastModifiedStr method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFileLastModifiedStrAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns the owner of a file. pathOrHandle may be a remote filepath or an open handle
	// string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle must be set to
	// true, otherwise it should be false. If bFollowLinks is true, then symbolic links
	// will be followed on the server.
	// 
	// Note: Servers running the SFTP v3 protocol or lower do not have the ability to
	// return a file's owner name. Instead, the decimal UID of the file is returned.
	// 
	bool GetFileOwner(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, CkString &outStr);
	// Returns the owner of a file. pathOrHandle may be a remote filepath or an open handle
	// string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle must be set to
	// true, otherwise it should be false. If bFollowLinks is true, then symbolic links
	// will be followed on the server.
	// 
	// Note: Servers running the SFTP v3 protocol or lower do not have the ability to
	// return a file's owner name. Instead, the decimal UID of the file is returned.
	// 
	const wchar_t *getFileOwner(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);
	// Returns the owner of a file. pathOrHandle may be a remote filepath or an open handle
	// string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle must be set to
	// true, otherwise it should be false. If bFollowLinks is true, then symbolic links
	// will be followed on the server.
	// 
	// Note: Servers running the SFTP v3 protocol or lower do not have the ability to
	// return a file's owner name. Instead, the decimal UID of the file is returned.
	// 
	const wchar_t *fileOwner(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFileOwner method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFileOwnerAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns the access permisssions flags of a file. pathOrHandle may be a remote filepath
	// or an open handle string as returned by OpenFile. If pathOrHandle is a handle, then bIsHandle
	// must be set to true, otherwise it should be false. If bFollowLinks is true, then
	// symbolic links will be followed on the server.
	int GetFilePermissions(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates an asynchronous task to call the GetFilePermissions method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetFilePermissionsAsync(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns the size in bytes of a file on the SSH server. If the file size exceeds
	// what can be represented in 32-bits, a value of -1 is returned. pathOrHandle may be a
	// remote filepath or an open handle string as returned by OpenFile. If pathOrHandle is a
	// handle, then bIsHandle must be set to true, otherwise it should be false. If bFollowLinks
	// is true, then symbolic links will be followed on the server.
	int GetFileSize32(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns a 64-bit integer containing the size (in bytes) of a file on the SSH
	// server. pathOrHandle may be a remote filepath or an open handle string as returned by
	// OpenFile. If pathOrHandle is a handle, then bIsHandle must be set to true, otherwise it
	// should be false. If bFollowLinks is true, then symbolic links will be followed on
	// the server.
	__int64 GetFileSize64(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Returns the size in bytes (in decimal string form) of a file on the SSH server.
	// pathOrHandle may be a remote filepath or an open handle string as returned by OpenFile.
	// If pathOrHandle is a handle, then bIsHandle must be set to true, otherwise it should be
	// false. If bFollowLinks is true, then symbolic links will be followed on the server.
	// 
	// Note: This method exists for environments that do not have 64-bit integer
	// support. The Add64 method is provided for 64-bit addition, and other methods
	// such as ReadFileBytes64s allow for 64-bit values to be passed as strings.
	// 
	bool GetFileSizeStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle, CkString &outStr);
	// Returns the size in bytes (in decimal string form) of a file on the SSH server.
	// pathOrHandle may be a remote filepath or an open handle string as returned by OpenFile.
	// If pathOrHandle is a handle, then bIsHandle must be set to true, otherwise it should be
	// false. If bFollowLinks is true, then symbolic links will be followed on the server.
	// 
	// Note: This method exists for environments that do not have 64-bit integer
	// support. The Add64 method is provided for 64-bit addition, and other methods
	// such as ReadFileBytes64s allow for 64-bit values to be passed as strings.
	// 
	const wchar_t *getFileSizeStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);
	// Returns the size in bytes (in decimal string form) of a file on the SSH server.
	// pathOrHandle may be a remote filepath or an open handle string as returned by OpenFile.
	// If pathOrHandle is a handle, then bIsHandle must be set to true, otherwise it should be
	// false. If bFollowLinks is true, then symbolic links will be followed on the server.
	// 
	// Note: This method exists for environments that do not have 64-bit integer
	// support. The Add64 method is provided for 64-bit addition, and other methods
	// such as ReadFileBytes64s allow for 64-bit values to be passed as strings.
	// 
	const wchar_t *fileSizeStr(const wchar_t *pathOrHandle, bool bFollowLinks, bool bIsHandle);

	// Creates a hard link on the server using the hardlink@openssh.com extension. This
	// only works for SFTP servers that support the hardlink@openssh.com extension.
	bool HardLink(const wchar_t *oldPath, const wchar_t *newPath);

	// Creates an asynchronous task to call the HardLink method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *HardLinkAsync(const wchar_t *oldPath, const wchar_t *newPath);

	// Intializes the SFTP subsystem. This should be called after connecting and
	// authenticating. An SFTP session always begins by first calling Connect to
	// connect to the SSH server, then calling either AuthenticatePw or AuthenticatePk
	// to login, and finally calling InitializeSftp.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	// If this method fails, the reason may be present in the InitializeFailCode and
	// InitializeFailReason properties (assuming the failure occurred when trying to
	// open the SFTP session channel).
	// 
	bool InitializeSftp(void);

	// Creates an asynchronous task to call the InitializeSftp method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *InitializeSftpAsync(void);

	// Provides information about what transpired in the last method called. For many
	// methods, there is no information. For some methods, details about what
	// transpired can be obtained via LastJsonData.
	// The caller is responsible for deleting the object returned by this method.
	CkJsonObjectW *LastJsonData(void);

	// Returns true if the last read on the specified handle failed. Otherwise
	// returns false.
	bool LastReadFailed(const wchar_t *handle);

	// Returns the number of bytes received by the last read on a specified channel.
	int LastReadNumBytes(const wchar_t *handle);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Opens a directory for reading. To get a directory listing, first open the
	// directory by calling this method, then call ReadDir to read the directory, and
	// finally call CloseHandle to close the directory.
	// 
	// The SFTP protocol represents file names as strings. File names are assumed to
	// use the slash ('/') character as a directory separator.
	// 
	// File names starting with a slash are "absolute", and are relative to the root of
	// the file system. Names starting with any other character are relative to the
	// user's default directory (home directory). Note that identifying the user is
	// assumed to take place outside of this protocol.
	// 
	// Servers SHOULD interpret a path name component ".." as referring to the parent
	// directory, and "." as referring to the current directory.
	// 
	// An empty path name is valid, and it refers to the user's default directory
	// (usually the user's home directory).
	// 
	// Please note: This method does NOT "change" the remote working directory. It is
	// only a method for opening a directory for the purpose of reading the directory
	// listing.
	// 
	// SFTP is Secure File Transfer over SSH. It is not the FTP protocol. There is no
	// similarity or relationship between FTP and SFTP. Therefore, concepts such as
	// "current remote directory" that exist in FTP do not exist with SFTP. With the
	// SFTP protocol, the current directory will always be the home directory of the
	// user account used during SSH/SFTP authentication. You may pass relative or
	// absolute directory/file paths. A relative path is always relative to the home
	// directory of the SSH user account.
	// 
	bool OpenDir(const wchar_t *path, CkString &outStr);
	// Opens a directory for reading. To get a directory listing, first open the
	// directory by calling this method, then call ReadDir to read the directory, and
	// finally call CloseHandle to close the directory.
	// 
	// The SFTP protocol represents file names as strings. File names are assumed to
	// use the slash ('/') character as a directory separator.
	// 
	// File names starting with a slash are "absolute", and are relative to the root of
	// the file system. Names starting with any other character are relative to the
	// user's default directory (home directory). Note that identifying the user is
	// assumed to take place outside of this protocol.
	// 
	// Servers SHOULD interpret a path name component ".." as referring to the parent
	// directory, and "." as referring to the current directory.
	// 
	// An empty path name is valid, and it refers to the user's default directory
	// (usually the user's home directory).
	// 
	// Please note: This method does NOT "change" the remote working directory. It is
	// only a method for opening a directory for the purpose of reading the directory
	// listing.
	// 
	// SFTP is Secure File Transfer over SSH. It is not the FTP protocol. There is no
	// similarity or relationship between FTP and SFTP. Therefore, concepts such as
	// "current remote directory" that exist in FTP do not exist with SFTP. With the
	// SFTP protocol, the current directory will always be the home directory of the
	// user account used during SSH/SFTP authentication. You may pass relative or
	// absolute directory/file paths. A relative path is always relative to the home
	// directory of the SSH user account.
	// 
	const wchar_t *openDir(const wchar_t *path);

	// Creates an asynchronous task to call the OpenDir method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpenDirAsync(const wchar_t *path);

	// Opens or creates a file on the remote system. Returns a handle which may be
	// passed to methods for reading and/or writing the file. The remotePath is the remote
	// file path (the path to the file on the server). When the application is finished
	// with the handle, it should call CloseHandle(remotePath).
	// 
	// access should be one of the following strings: "readOnly", "writeOnly", or
	// "readWrite".
	// 
	// createDisposition is a comma-separated list of keywords to provide more control over how the
	// file is opened or created. One of the following keywords must be present:
	// "createNew", "createTruncate", "openExisting", "openOrCreate", or
	// "truncateExisting". All other keywords are optional. The list of keywords and
	// their meanings are shown here:
	// 
	// createNew
	// A new file is created; if the file already exists the method fails.
	// 
	// createTruncate
	// A new file is created; if the file already exists, it is opened and truncated.
	// 
	// openExisting
	// An existing file is opened. If the file does not exist the method fails.
	// 
	// openOrCreate
	// If the file exists, it is opened. If the file does not exist, it is created.
	// 
	// truncateExisting
	// An existing file is opened and truncated. If the file does not exist the method
	// fails.
	// 
	// appendData
	// Data is always written at the end of the file. Data is not required to be
	// appended atomically. This means that if multiple writers attempt to append data
	// simultaneously, data from the first may be lost.
	// 
	// appendDataAtomic
	// Data is always written at the end of the file. Data MUST be written atomically
	// so that there is no chance that multiple appenders can collide and result in
	// data being lost.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// textMode
	// Indicates that the server should treat the file as text and convert it to the
	// canonical newline convention in use. When a file is opened with this flag, data
	// is always appended to the end of the file. Servers MUST process multiple,
	// parallel reads and writes correctly in this mode.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// blockRead
	// The server MUST guarantee that no other handle has been opened with read access,
	// and that no other handle will be opened with read access until the client closes
	// the handle. (This MUST apply both to other clients and to other processes on the
	// server.) In a nutshell, this opens the file in non-sharing mode.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// blockWrite
	// The server MUST guarantee that no other handle has been opened with write
	// access, and that no other handle will be opened with write access until the
	// client closes the handle. (This MUST apply both to other clients and to other
	// processes on the server.) In a nutshell, this opens the file in non-sharing
	// mode.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// blockDelete
	// The server MUST guarantee that the file itself is not deleted in any other way
	// until the client closes the handle. No other client or process is allowed to
	// open the file with delete access.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// blockAdvisory
	// If set, the above "block" modes are advisory. In advisory mode, only other
	// accesses that specify a "block" mode need be considered when determining whether
	// the "block" can be granted, and the server need not prevent I/O operations that
	// violate the block mode. The server MAY perform mandatory locking even if the
	// blockAdvisory flag is set.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// noFollow
	// If the final component of the path is a symlink, then the open MUST fail.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// deleteOnClose
	// The file should be deleted when the last handle to it is closed. (The last
	// handle may not be an sftp-handle.) This MAY be emulated by a server if the OS
	// doesn't support it by deleting the file when this handle is closed.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// accessAuditAlarmInfo
	// The client wishes the server to enable any privileges or extra capabilities that
	// the user may have in to allow the reading and writing of AUDIT or ALARM access
	// control entries.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// accessBackup
	// The client wishes the server to enable any privileges or extra capabilities that
	// the user may have in order to bypass normal access checks for the purpose of
	// backing up or restoring files.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// backupStream
	// This flag indicates that the client wishes to read or write a backup stream. A
	// backup stream is a system dependent structured data stream that encodes all the
	// information that must be preserved in order to restore the file from backup
	// medium. The only well defined use for backup stream data read in this fashion is
	// to write it to the same server to a file also opened using the backupStream
	// flag. However, if the server has a well defined backup stream format, there may
	// be other uses for this data outside the scope of this protocol.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// IMPORANT: If remotePath is a filename with no path, such as "test.txt", and the server
	// responds with a "Folder not found" error, then try prepending "./" to the remotePath.
	// For example, instead of passing "test.txt", try "./test.txt".
	// 
	// IMPORTANT note about createDisposition: Many of the options, such as textMode, are not
	// implemented in the SFTP protocol versions 3 and 4. Only SFTP servers at protocol
	// version 5 or later support these options. You can find out the protocol version
	// of your server by examining the value of the ProtocolVersion property after
	// calling InitializeSftp. Also, make sure the ForceV3 property is set to false
	// (the default value is true)
	// 
	bool OpenFile(const wchar_t *remotePath, const wchar_t *access, const wchar_t *createDisposition, CkString &outStr);
	// Opens or creates a file on the remote system. Returns a handle which may be
	// passed to methods for reading and/or writing the file. The remotePath is the remote
	// file path (the path to the file on the server). When the application is finished
	// with the handle, it should call CloseHandle(remotePath).
	// 
	// access should be one of the following strings: "readOnly", "writeOnly", or
	// "readWrite".
	// 
	// createDisposition is a comma-separated list of keywords to provide more control over how the
	// file is opened or created. One of the following keywords must be present:
	// "createNew", "createTruncate", "openExisting", "openOrCreate", or
	// "truncateExisting". All other keywords are optional. The list of keywords and
	// their meanings are shown here:
	// 
	// createNew
	// A new file is created; if the file already exists the method fails.
	// 
	// createTruncate
	// A new file is created; if the file already exists, it is opened and truncated.
	// 
	// openExisting
	// An existing file is opened. If the file does not exist the method fails.
	// 
	// openOrCreate
	// If the file exists, it is opened. If the file does not exist, it is created.
	// 
	// truncateExisting
	// An existing file is opened and truncated. If the file does not exist the method
	// fails.
	// 
	// appendData
	// Data is always written at the end of the file. Data is not required to be
	// appended atomically. This means that if multiple writers attempt to append data
	// simultaneously, data from the first may be lost.
	// 
	// appendDataAtomic
	// Data is always written at the end of the file. Data MUST be written atomically
	// so that there is no chance that multiple appenders can collide and result in
	// data being lost.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// textMode
	// Indicates that the server should treat the file as text and convert it to the
	// canonical newline convention in use. When a file is opened with this flag, data
	// is always appended to the end of the file. Servers MUST process multiple,
	// parallel reads and writes correctly in this mode.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// blockRead
	// The server MUST guarantee that no other handle has been opened with read access,
	// and that no other handle will be opened with read access until the client closes
	// the handle. (This MUST apply both to other clients and to other processes on the
	// server.) In a nutshell, this opens the file in non-sharing mode.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// blockWrite
	// The server MUST guarantee that no other handle has been opened with write
	// access, and that no other handle will be opened with write access until the
	// client closes the handle. (This MUST apply both to other clients and to other
	// processes on the server.) In a nutshell, this opens the file in non-sharing
	// mode.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// blockDelete
	// The server MUST guarantee that the file itself is not deleted in any other way
	// until the client closes the handle. No other client or process is allowed to
	// open the file with delete access.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// blockAdvisory
	// If set, the above "block" modes are advisory. In advisory mode, only other
	// accesses that specify a "block" mode need be considered when determining whether
	// the "block" can be granted, and the server need not prevent I/O operations that
	// violate the block mode. The server MAY perform mandatory locking even if the
	// blockAdvisory flag is set.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// noFollow
	// If the final component of the path is a symlink, then the open MUST fail.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// deleteOnClose
	// The file should be deleted when the last handle to it is closed. (The last
	// handle may not be an sftp-handle.) This MAY be emulated by a server if the OS
	// doesn't support it by deleting the file when this handle is closed.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// accessAuditAlarmInfo
	// The client wishes the server to enable any privileges or extra capabilities that
	// the user may have in to allow the reading and writing of AUDIT or ALARM access
	// control entries.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// accessBackup
	// The client wishes the server to enable any privileges or extra capabilities that
	// the user may have in order to bypass normal access checks for the purpose of
	// backing up or restoring files.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// backupStream
	// This flag indicates that the client wishes to read or write a backup stream. A
	// backup stream is a system dependent structured data stream that encodes all the
	// information that must be preserved in order to restore the file from backup
	// medium. The only well defined use for backup stream data read in this fashion is
	// to write it to the same server to a file also opened using the backupStream
	// flag. However, if the server has a well defined backup stream format, there may
	// be other uses for this data outside the scope of this protocol.
	// (Only supported in SFTP protocol versions 5 and later. See the note below.)
	// 
	// IMPORANT: If remotePath is a filename with no path, such as "test.txt", and the server
	// responds with a "Folder not found" error, then try prepending "./" to the remotePath.
	// For example, instead of passing "test.txt", try "./test.txt".
	// 
	// IMPORTANT note about createDisposition: Many of the options, such as textMode, are not
	// implemented in the SFTP protocol versions 3 and 4. Only SFTP servers at protocol
	// version 5 or later support these options. You can find out the protocol version
	// of your server by examining the value of the ProtocolVersion property after
	// calling InitializeSftp. Also, make sure the ForceV3 property is set to false
	// (the default value is true)
	// 
	const wchar_t *openFile(const wchar_t *remotePath, const wchar_t *access, const wchar_t *createDisposition);

	// Creates an asynchronous task to call the OpenFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpenFileAsync(const wchar_t *remotePath, const wchar_t *access, const wchar_t *createDisposition);

	// Reads the contents of a directory and returns the directory listing (as an
	// object). The handle returned by OpenDir should be passed to this method.
	// The caller is responsible for deleting the object returned by this method.
	CkSFtpDirW *ReadDir(const wchar_t *handle);

	// Creates an asynchronous task to call the ReadDir method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadDirAsync(const wchar_t *handle);

	// Reads file data from a remote file on the SSH server. The handle is a file handle
	// returned by the OpenFile method. The numBytes is the maximum number of bytes to
	// read. If the end-of-file is reached prior to reading the number of requested
	// bytes, then fewer bytes may be returned. The received bytes are appended to the
	// contents of bd.
	// 
	// To read an entire file, one may call ReadFileBd repeatedly until Eof(handle)
	// returns true.
	// 
	bool ReadFileBd(const wchar_t *handle, int numBytes, CkBinDataW &bd);

	// Creates an asynchronous task to call the ReadFileBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadFileBdAsync(const wchar_t *handle, int numBytes, CkBinDataW &bd);

	// Reads file data from a remote file on the SSH server. The handle is a file handle
	// returned by the OpenFile method. The numBytes is the maximum number of bytes to
	// read. If the end-of-file is reached prior to reading the number of requested
	// bytes, then fewer bytes may be returned.
	// 
	// To read an entire file, one may call ReadFileBytes repeatedly until Eof(handle)
	// returns true.
	// 
	bool ReadFileBytes(const wchar_t *handle, int numBytes, CkByteData &outBytes);

	// Creates an asynchronous task to call the ReadFileBytes method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadFileBytesAsync(const wchar_t *handle, int numBytes);

	// Reads file data from a remote file on the SSH server. The handle is a file handle
	// returned by the OpenFile method. The offset is measured in bytes relative to the
	// beginning of the file. (64-bit offsets are supported via the ReadFileBytes64 and
	// ReadFileBytes64s methods.) The offset is ignored if the "textMode" flag was
	// specified during the OpenFile. The numBytes is the maximum number of bytes to read.
	// If the end-of-file is reached prior to reading the number of requested bytes,
	// then fewer bytes may be returned.
	bool ReadFileBytes32(const wchar_t *handle, int offset, int numBytes, CkByteData &outBytes);

	// Reads file data from a remote file on the SSH server. The handle is a file handle
	// returned by the OpenFile method. The offset is a 64-bit integer measured in bytes
	// relative to the beginning of the file. The offset is ignored if the "textMode"
	// flag was specified during the OpenFile. The numBytes is the maximum number of bytes
	// to read. If the end-of-file is reached prior to reading the number of requested
	// bytes, then fewer bytes may be returned.
	bool ReadFileBytes64(const wchar_t *handle, __int64 offset, int numBytes, CkByteData &outBytes);

	// (This method exists for systems that do not support 64-bit integers. The 64-bit
	// integer offset is passed as a decimal string instead.)
	// 
	// Reads file data from a remote file on the SSH server. The handle is a file handle
	// returned by the OpenFile method. The offset is a 64-bit integer represented as a
	// decimal string. It represents an offset in bytes from the beginning of the file.
	// The offset is ignored if the "textMode" flag was specified during the OpenFile.
	// The numBytes is the maximum number of bytes to read. If the end-of-file is reached
	// prior to reading the number of requested bytes, then fewer bytes may be
	// returned.
	// 
	bool ReadFileBytes64s(const wchar_t *handle, const wchar_t *offset, int numBytes, CkByteData &outBytes);

	// This method is identical to ReadFileBytes except for one thing: The bytes are
	// interpreted according to the specified charset (i.e. the character encoding) and
	// returned as a string. A list of supported charset values may be found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: If the charset is an encoding where a single character might be represented
	// in multiple bytes (such as utf-8, Shift_JIS, etc.) then there is a risk that the
	// very last character may be partially read. This is because the method specifies
	// the number of bytes to read, not the number of characters. This is never a
	// problem with character encodings that use a single byte per character, such as
	// all of the iso-8859-* encodings, or the Windows-* encodings.
	// 
	// To read an entire file, one may call ReadFileText repeatedly until Eof(handle)
	// returns true.
	// 
	bool ReadFileText(const wchar_t *handle, int numBytes, const wchar_t *charset, CkString &outStr);
	// This method is identical to ReadFileBytes except for one thing: The bytes are
	// interpreted according to the specified charset (i.e. the character encoding) and
	// returned as a string. A list of supported charset values may be found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: If the charset is an encoding where a single character might be represented
	// in multiple bytes (such as utf-8, Shift_JIS, etc.) then there is a risk that the
	// very last character may be partially read. This is because the method specifies
	// the number of bytes to read, not the number of characters. This is never a
	// problem with character encodings that use a single byte per character, such as
	// all of the iso-8859-* encodings, or the Windows-* encodings.
	// 
	// To read an entire file, one may call ReadFileText repeatedly until Eof(handle)
	// returns true.
	// 
	const wchar_t *readFileText(const wchar_t *handle, int numBytes, const wchar_t *charset);

	// Creates an asynchronous task to call the ReadFileText method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadFileTextAsync(const wchar_t *handle, int numBytes, const wchar_t *charset);

	// This method is identical to ReadFileBytes32 except for one thing: The bytes are
	// interpreted according to the specified charset (i.e. the character encoding) and
	// returned as a string. A list of supported charset values may be found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: If the charset is an encoding where a single character might be represented
	// in multiple bytes (such as utf-8, Shift_JIS, etc.) then there is a risk that the
	// very last character may be partially read. This is because the method specifies
	// the number of bytes to read, not the number of characters. This is never a
	// problem with character encodings that use a single byte per character, such as
	// all of the iso-8859-* encodings, or the Windows-* encodings.
	// 
	bool ReadFileText32(const wchar_t *handle, int offset, int numBytes, const wchar_t *charset, CkString &outStr);
	// This method is identical to ReadFileBytes32 except for one thing: The bytes are
	// interpreted according to the specified charset (i.e. the character encoding) and
	// returned as a string. A list of supported charset values may be found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: If the charset is an encoding where a single character might be represented
	// in multiple bytes (such as utf-8, Shift_JIS, etc.) then there is a risk that the
	// very last character may be partially read. This is because the method specifies
	// the number of bytes to read, not the number of characters. This is never a
	// problem with character encodings that use a single byte per character, such as
	// all of the iso-8859-* encodings, or the Windows-* encodings.
	// 
	const wchar_t *readFileText32(const wchar_t *handle, int offset, int numBytes, const wchar_t *charset);

	// This method is identical to ReadFileBytes64 except for one thing: The bytes are
	// interpreted according to the specified charset (i.e. the character encoding) and
	// returned as a string. A list of supported charset values may be found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: If the charset is an encoding where a single character might be represented
	// in multiple bytes (such as utf-8, Shift_JIS, etc.) then there is a risk that the
	// very last character may be partially read. This is because the method specifies
	// the number of bytes to read, not the number of characters. This is never a
	// problem with character encodings that use a single byte per character, such as
	// all of the iso-8859-* encodings, or the Windows-* encodings.
	// 
	bool ReadFileText64(const wchar_t *handle, __int64 offset, int numBytes, const wchar_t *charset, CkString &outStr);
	// This method is identical to ReadFileBytes64 except for one thing: The bytes are
	// interpreted according to the specified charset (i.e. the character encoding) and
	// returned as a string. A list of supported charset values may be found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: If the charset is an encoding where a single character might be represented
	// in multiple bytes (such as utf-8, Shift_JIS, etc.) then there is a risk that the
	// very last character may be partially read. This is because the method specifies
	// the number of bytes to read, not the number of characters. This is never a
	// problem with character encodings that use a single byte per character, such as
	// all of the iso-8859-* encodings, or the Windows-* encodings.
	// 
	const wchar_t *readFileText64(const wchar_t *handle, __int64 offset, int numBytes, const wchar_t *charset);

	// This method is identical to ReadFileBytes64s except for one thing: The bytes are
	// interpreted according to the specified charset (i.e. the character encoding) and
	// returned as a string. A list of supported charset values may be found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: If the charset is an encoding where a single character might be represented
	// in multiple bytes (such as utf-8, Shift_JIS, etc.) then there is a risk that the
	// very last character may be partially read. This is because the method specifies
	// the number of bytes to read, not the number of characters. This is never a
	// problem with character encodings that use a single byte per character, such as
	// all of the iso-8859-* encodings, or the Windows-* encodings.
	// 
	bool ReadFileText64s(const wchar_t *handle, const wchar_t *offset, int numBytes, const wchar_t *charset, CkString &outStr);
	// This method is identical to ReadFileBytes64s except for one thing: The bytes are
	// interpreted according to the specified charset (i.e. the character encoding) and
	// returned as a string. A list of supported charset values may be found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: If the charset is an encoding where a single character might be represented
	// in multiple bytes (such as utf-8, Shift_JIS, etc.) then there is a risk that the
	// very last character may be partially read. This is because the method specifies
	// the number of bytes to read, not the number of characters. This is never a
	// problem with character encodings that use a single byte per character, such as
	// all of the iso-8859-* encodings, or the Windows-* encodings.
	// 
	const wchar_t *readFileText64s(const wchar_t *handle, const wchar_t *offset, int numBytes, const wchar_t *charset);

	// Returns the target of a symbolic link on the server. The path is the path of the
	// symbolic link on the server.
	bool ReadLink(const wchar_t *path, CkString &outStr);
	// Returns the target of a symbolic link on the server. The path is the path of the
	// symbolic link on the server.
	const wchar_t *readLink(const wchar_t *path);

	// Creates an asynchronous task to call the ReadLink method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadLinkAsync(const wchar_t *path);

	// This method can be used to have the server canonicalize any given path name to
	// an absolute path. This is useful for converting path names containing ".."
	// components or relative pathnames without a leading slash into absolute paths.
	// The absolute path is returned by this method.
	// 
	// originalPath is the first component of the path which the client wishes resolved into a
	// absolute canonical path. This may be the entire path.
	// 
	// The composePath is a path which the client wishes the server to compose with the
	// original path to form the new path. This field is optional and may be set to a
	// zero-length string.
	// 
	// The server will take the originalPath and apply the composePath as a modification to it. composePath
	// may be relative to originalPath or may be an absolute path, in which case originalPath will be
	// discarded. The composePath may be zero length.
	// 
	// Note: Servers running SFTP v4 and below do not support composePath.
	// 
	bool RealPath(const wchar_t *originalPath, const wchar_t *composePath, CkString &outStr);
	// This method can be used to have the server canonicalize any given path name to
	// an absolute path. This is useful for converting path names containing ".."
	// components or relative pathnames without a leading slash into absolute paths.
	// The absolute path is returned by this method.
	// 
	// originalPath is the first component of the path which the client wishes resolved into a
	// absolute canonical path. This may be the entire path.
	// 
	// The composePath is a path which the client wishes the server to compose with the
	// original path to form the new path. This field is optional and may be set to a
	// zero-length string.
	// 
	// The server will take the originalPath and apply the composePath as a modification to it. composePath
	// may be relative to originalPath or may be an absolute path, in which case originalPath will be
	// discarded. The composePath may be zero length.
	// 
	// Note: Servers running SFTP v4 and below do not support composePath.
	// 
	const wchar_t *realPath(const wchar_t *originalPath, const wchar_t *composePath);

	// Creates an asynchronous task to call the RealPath method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *RealPathAsync(const wchar_t *originalPath, const wchar_t *composePath);

	// Deletes a directory on the remote server. Most (if not all) SFTP servers require
	// that the directorybe empty of files before it may be deleted.
	bool RemoveDir(const wchar_t *path);

	// Creates an asynchronous task to call the RemoveDir method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *RemoveDirAsync(const wchar_t *path);

	// Deletes a file on the SFTP server.
	bool RemoveFile(const wchar_t *filename);

	// Creates an asynchronous task to call the RemoveFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *RemoveFileAsync(const wchar_t *filename);

	// Renames a file or directory on the SFTP server. This is the equivalent of moving
	// a file. To move a file from one directory to another, call this method where
	// oldPath is the path of the existing file, and newPath is the path where the file is to
	// be moved. For example, rename a file from "someDirA/filename" to
	// "someDirB/abc/xyz/filename".
	bool RenameFileOrDir(const wchar_t *oldPath, const wchar_t *newPath);

	// Creates an asynchronous task to call the RenameFileOrDir method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *RenameFileOrDirAsync(const wchar_t *oldPath, const wchar_t *newPath);

	// Resumes an SFTP download. The size of the localFilePath is checked and the download
	// begins at the appropriate position in the remoteFilePath. If localFilePath is empty or
	// non-existent, then this method is identical to DownloadFileByName. If the localFilePath
	// is already fully downloaded, then no additional data is downloaded and the
	// method will return true.
	bool ResumeDownloadFileByName(const wchar_t *remoteFilePath, const wchar_t *localFilePath);

	// Creates an asynchronous task to call the ResumeDownloadFileByName method with
	// the arguments provided. (Async methods are available starting in Chilkat
	// v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ResumeDownloadFileByNameAsync(const wchar_t *remoteFilePath, const wchar_t *localFilePath);

	// Resumes a file upload to the SFTP/SSH server. The size of the remoteFilePath is first
	// checked to determine the starting offset for the upload. If remoteFilePath is empty or
	// does not exist, this method is equivalent to UploadFileByName. If remoteFilePath is
	// already fully uploaded (i.e. it's size is equal to localFilePath), then no additional
	// bytes are uploaded and true is returned.
	bool ResumeUploadFileByName(const wchar_t *remoteFilePath, const wchar_t *localFilePath);

	// Creates an asynchronous task to call the ResumeUploadFileByName method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ResumeUploadFileByNameAsync(const wchar_t *remoteFilePath, const wchar_t *localFilePath);

	// Sends an IGNORE message to the SSH server. This is one way of verifying that the
	// connection to the SSH server is open and valid. The SSH server does not respond
	// to an IGNORE message. It simply ignores it. IGNORE messages are not associated
	// with a channel (i.e., you do not need to first open a channel prior to sending
	// an IGNORE message).
	bool SendIgnore(void);

	// Creates an asynchronous task to call the SendIgnore method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendIgnoreAsync(void);

	// Sets the create date/time for a file on the server. The pathOrHandle may be a filepath
	// or the handle of a currently open file. isHandle should be set to true if the pathOrHandle
	// is a handle, otherwise set isHandle to false.
	// 
	// Note: Servers running version 3 or lower of the SFTP protocol do not support
	// setting the create date/time.
	// 
	bool SetCreateDt(const wchar_t *pathOrHandle, bool isHandle, CkDateTimeW &createDateTime);

	// Creates an asynchronous task to call the SetCreateDt method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetCreateDtAsync(const wchar_t *pathOrHandle, bool isHandle, CkDateTimeW &createDateTime);

	// Sets the create date/time for a file on the server. The pathOrHandle may be a filepath
	// or the handle of a currently open file. isHandle should be set to true if the pathOrHandle
	// is a handle, otherwise set isHandle to false.
	// 
	// Note: Servers running version 3 or lower of the SFTP protocol do not support
	// setting the create date/time.
	// 
	bool SetCreateTime(const wchar_t *pathOrHandle, bool isHandle, SYSTEMTIME &createDateTime);

	// The same as SetCreateTime, except the date/time is passed as an RFC822 formatted
	// string.
	bool SetCreateTimeStr(const wchar_t *pathOrHandle, bool bIsHandle, const wchar_t *dateTimeStr);

	// Creates an asynchronous task to call the SetCreateTimeStr method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetCreateTimeStrAsync(const wchar_t *pathOrHandle, bool bIsHandle, const wchar_t *dateTimeStr);

	// Sets the last-access date/time for a file on the server. The pathOrHandle may be a
	// filepath or the handle of a currently open file. isHandle should be set to true if
	// the pathOrHandle is a handle, otherwise set isHandle to false.
	bool SetLastAccessDt(const wchar_t *pathOrHandle, bool isHandle, CkDateTimeW &accessDateTime);

	// Creates an asynchronous task to call the SetLastAccessDt method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetLastAccessDtAsync(const wchar_t *pathOrHandle, bool isHandle, CkDateTimeW &accessDateTime);

	// Sets the last-access date/time for a file on the server. The pathOrHandle may be a
	// filepath or the handle of a currently open file. isHandle should be set to true if
	// the pathOrHandle is a handle, otherwise set isHandle to false.
	bool SetLastAccessTime(const wchar_t *pathOrHandle, bool isHandle, SYSTEMTIME &accessDateTime);

	// The same as SetLastAccessTime, except the date/time is passed as an RFC822
	// formatted string.
	bool SetLastAccessTimeStr(const wchar_t *pathOrHandle, bool bIsHandle, const wchar_t *dateTimeStr);

	// Creates an asynchronous task to call the SetLastAccessTimeStr method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetLastAccessTimeStrAsync(const wchar_t *pathOrHandle, bool bIsHandle, const wchar_t *dateTimeStr);

	// Sets the last-modified date/time for a file on the server. The pathOrHandle may be a
	// filepath or the handle of a currently open file. isHandle should be set to true if
	// the pathOrHandle is a handle, otherwise set isHandle to false.
	bool SetLastModifiedDt(const wchar_t *pathOrHandle, bool isHandle, CkDateTimeW &modifiedDateTime);

	// Creates an asynchronous task to call the SetLastModifiedDt method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetLastModifiedDtAsync(const wchar_t *pathOrHandle, bool isHandle, CkDateTimeW &modifiedDateTime);

	// Sets the last-modified date/time for a file on the server. The pathOrHandle may be a
	// filepath or the handle of a currently open file. isHandle should be set to true if
	// the pathOrHandle is a handle, otherwise set isHandle to false.
	bool SetLastModifiedTime(const wchar_t *pathOrHandle, bool isHandle, SYSTEMTIME &modifiedDateTime);

	// The same as SetLastModifiedTime, except the date/time is passed as an RFC822
	// formatted string.
	bool SetLastModifiedTimeStr(const wchar_t *pathOrHandle, bool bIsHandle, const wchar_t *dateTimeStr);

	// Creates an asynchronous task to call the SetLastModifiedTimeStr method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetLastModifiedTimeStrAsync(const wchar_t *pathOrHandle, bool bIsHandle, const wchar_t *dateTimeStr);

	// Sets the owner and group for a file on the server. The pathOrHandle may be a filepath or
	// the handle of a currently open file. isHandle should be set to true if the pathOrHandle is
	// a handle, otherwise set isHandle to false.
	// 
	// Note: Servers running version 3 or lower of the SFTP protocol do not support
	// setting the owner and group.
	// 
	bool SetOwnerAndGroup(const wchar_t *pathOrHandle, bool isHandle, const wchar_t *owner, const wchar_t *group);

	// Creates an asynchronous task to call the SetOwnerAndGroup method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetOwnerAndGroupAsync(const wchar_t *pathOrHandle, bool isHandle, const wchar_t *owner, const wchar_t *group);

	// Sets the permissions for a file on the server. The pathOrHandle may be a filepath or the
	// handle of a currently open file. isHandle should be set to true if the pathOrHandle is a
	// handle, otherwise set isHandle to false.
	bool SetPermissions(const wchar_t *pathOrHandle, bool isHandle, int permissions);

	// Creates an asynchronous task to call the SetPermissions method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SetPermissionsAsync(const wchar_t *pathOrHandle, bool isHandle, int permissions);

	// Begins keyboard-interactive authentication with the SSH server. Returns an XML
	// string providing the name, instruction, and prompts. The XML has the following
	// format:
	//  	_LT_infoRequest numPrompts="N"_GT_
	// 	    _LT_name_GT_name_string_LT_/name_GT_
	// 	    _LT_instruction_GT_instruction_string_LT_/instruction_GT_
	// 	    _LT_prompt1 echo="1_or_0"_GT_prompt_string_LT_/prompt1_GT_
	// 	    ...
	// 	    _LT_promptN echo="1_or_0"_GT_prompt_string_LT_/promptN_GT_
	// 	_LT_/infoRequest_GT_
	// 
	// If the authentication immediately succeeds because no password is required, or
	// immediately fails, the XML response can be:
	// _LT_success_GT_success_message_LT_/success_GT_
	// 
	// or
	// 
	// _LT_error_GT_error_message_LT_/error_GT_
	// 
	bool StartKeyboardAuth(const wchar_t *login, CkString &outStr);
	// Begins keyboard-interactive authentication with the SSH server. Returns an XML
	// string providing the name, instruction, and prompts. The XML has the following
	// format:
	//  	_LT_infoRequest numPrompts="N"_GT_
	// 	    _LT_name_GT_name_string_LT_/name_GT_
	// 	    _LT_instruction_GT_instruction_string_LT_/instruction_GT_
	// 	    _LT_prompt1 echo="1_or_0"_GT_prompt_string_LT_/prompt1_GT_
	// 	    ...
	// 	    _LT_promptN echo="1_or_0"_GT_prompt_string_LT_/promptN_GT_
	// 	_LT_/infoRequest_GT_
	// 
	// If the authentication immediately succeeds because no password is required, or
	// immediately fails, the XML response can be:
	// _LT_success_GT_success_message_LT_/success_GT_
	// 
	// or
	// 
	// _LT_error_GT_error_message_LT_/error_GT_
	// 
	const wchar_t *startKeyboardAuth(const wchar_t *login);

	// Creates an asynchronous task to call the StartKeyboardAuth method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *StartKeyboardAuthAsync(const wchar_t *login);

	// Create a symbolic link from oldpath to newpath on the server filesystem.
	bool SymLink(const wchar_t *oldPath, const wchar_t *newPath);

	// Creates an asynchronous task to call the SymLink method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SymLinkAsync(const wchar_t *oldPath, const wchar_t *newPath);

	// Downloads files from the SFTP server to a local directory tree. Synchronization
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
	//     
	// 
	// If recurse is false, then the remotel directory tree is not recursively
	// descended.
	// 
	bool SyncTreeDownload(const wchar_t *remoteRoot, const wchar_t *localRoot, int mode, bool recurse);

	// Creates an asynchronous task to call the SyncTreeDownload method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SyncTreeDownloadAsync(const wchar_t *remoteRoot, const wchar_t *localRoot, int mode, bool recurse);

	// Uploads a directory tree from the local filesystem to the SFTP server.
	// Synchronization modes include:
	// 
	//     mode=0: Upload all files
	//     mode=1: Upload all files that do not exist on the server.
	//     mode=2: Upload newer or non-existant files.
	//     mode=3: Upload only newer files. If a file does not already exist on the
	//     server, it is not uploaded.
	//     mode=4: transfer missing files or files with size differences.
	//     mode=5: same as mode 4, but also newer files.
	// 
	// If bRecurse is false, then the local directory tree is not recursively descended.
	bool SyncTreeUpload(const wchar_t *localBaseDir, const wchar_t *remoteBaseDir, int mode, bool bRecurse);

	// Creates an asynchronous task to call the SyncTreeUpload method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SyncTreeUploadAsync(const wchar_t *localBaseDir, const wchar_t *remoteBaseDir, int mode, bool bRecurse);

	// Unlocks the component. This must be called once prior to calling any other
	// method. A fully-functional 30-day trial is automatically started when an
	// arbitrary string is passed to this method. For example, passing "Hello", or
	// "abc123" will unlock the component for the 1st thirty days after the initial
	// install.
	// 
	// A purchased unlock code for SFTP should contain the substring ".SS" or "SSH" (or
	// it can be a Bundle unlock code) because SFTP is the Secure File Transfer
	// protocol over SSH. It is a sub-system of the SSH protocol. It is not the FTP
	// protocol. If the Chilkat FTP2 component/library should be used for the FTP
	// protocol.
	// 
	bool UnlockComponent(const wchar_t *unlockCode);

	// Uploads the contents of a BinData to a remote file.
	bool UploadBd(CkBinDataW &binData, const wchar_t *remoteFilePath);

	// Creates an asynchronous task to call the UploadBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UploadBdAsync(CkBinDataW &binData, const wchar_t *remoteFilePath);

	// Uploads a file from the local filesystem to the SFTP server. handle is a handle of
	// a currently open file (obtained by calling the OpenFile method). fromLocalFilePath is the
	// local file path of the file to be uploaded.
	bool UploadFile(const wchar_t *handle, const wchar_t *fromLocalFilePath);

	// Creates an asynchronous task to call the UploadFile method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UploadFileAsync(const wchar_t *handle, const wchar_t *fromLocalFilePath);

	// Simplified method for uploading a file to the SFTP/SSH server.
	// 
	// The last-modified date/time is only preserved if the PreserveDate property is
	// set to true. This behavior of maintaining the last-mod date/time was
	// introduced in v9.5.0.40.
	// 
	bool UploadFileByName(const wchar_t *remoteFilePath, const wchar_t *localFilePath);

	// Creates an asynchronous task to call the UploadFileByName method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UploadFileByNameAsync(const wchar_t *remoteFilePath, const wchar_t *localFilePath);

	// Uploads the contents of a StringBuilder to a remote file.
	bool UploadSb(CkStringBuilderW &sb, const wchar_t *remoteFilePath, const wchar_t *charset, bool includeBom);

	// Creates an asynchronous task to call the UploadSb method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *UploadSbAsync(CkStringBuilderW &sb, const wchar_t *remoteFilePath, const wchar_t *charset, bool includeBom);

	// Appends the contents of bd to an open file. The handle is a file handle returned
	// by the OpenFile method.
	bool WriteFileBd(const wchar_t *handle, CkBinDataW &bd);

	// Creates an asynchronous task to call the WriteFileBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteFileBdAsync(const wchar_t *handle, CkBinDataW &bd);

	// Appends byte data to an open file. The handle is a file handle returned by the
	// OpenFile method.
	bool WriteFileBytes(const wchar_t *handle, CkByteData &byteData);

	// Creates an asynchronous task to call the WriteFileBytes method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteFileBytesAsync(const wchar_t *handle, CkByteData &byteData);

	// Writes data to an open file at a specific offset from the beginning of the file.
	// The handle is a file handle returned by the OpenFile method. The offset is an offset
	// from the beginning of the file.
	bool WriteFileBytes32(const wchar_t *handle, int offset, CkByteData &data);

	// Writes data to an open file at a specific offset from the beginning of the file.
	// The handle is a file handle returned by the OpenFile method. The offset64 is an offset
	// from the beginning of the file.
	bool WriteFileBytes64(const wchar_t *handle, __int64 offset64, CkByteData &data);

	// Writes data to an open file at a specific offset from the beginning of the file.
	// The handle is a file handle returned by the OpenFile method. The offset64 is an offset
	// (in decimal string format) from the beginning of the file.
	bool WriteFileBytes64s(const wchar_t *handle, const wchar_t *offset64, CkByteData &data);

	// Appends character data to an open file. The handle is a file handle returned by
	// the OpenFile method. charset is a character encoding and is typically set to values
	// such as "ansi", "utf-8", "windows-1252", etc. A list of supported character
	// encodings is found on this page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// Note: It is necessary to specify the character encoding because in many
	// programming languages, strings are represented as Unicode (2 bytes/char) and in
	// most cases one does not wish to write Unicode chars to a text file (although it
	// is possible by setting charset = "Unicode").
	// 
	bool WriteFileText(const wchar_t *handle, const wchar_t *charset, const wchar_t *textData);

	// Creates an asynchronous task to call the WriteFileText method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteFileTextAsync(const wchar_t *handle, const wchar_t *charset, const wchar_t *textData);

	// Writes character data to an open file at a specific offset from the beginning of
	// the file. The handle is a file handle returned by the OpenFile method. charset is a
	// character encoding and is typically set to values such as "ansi", "utf-8",
	// "windows-1252", etc. A list of supported character encodings is found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	bool WriteFileText32(const wchar_t *handle, int offset32, const wchar_t *charset, const wchar_t *textData);

	// Writes character data to an open file at a specific offset from the beginning of
	// the file. The handle is a file handle returned by the OpenFile method. charset is a
	// character encoding and is typically set to values such as "ansi", "utf-8",
	// "windows-1252", etc. A list of supported character encodings is found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	bool WriteFileText64(const wchar_t *handle, __int64 offset64, const wchar_t *charset, const wchar_t *textData);

	// Writes character data to an open file at a specific offset from the beginning of
	// the file. The handle is a file handle returned by the OpenFile method. The offset64 is
	// an offset (in decimal string format) from the beginning of the file. charset is a
	// character encoding and is typically set to values such as "ansi", "utf-8",
	// "windows-1252", etc. A list of supported character encodings is found on this
	// page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	bool WriteFileText64s(const wchar_t *handle, const wchar_t *offset64, const wchar_t *charset, const wchar_t *textData);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
