// CkSshW.h: interface for the CkSshW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkSshW_H
#define _CkSshW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkSshKeyW;
class CkTaskW;
class CkSecureStringW;
class CkStringArrayW;
class CkByteData;
class CkJsonObjectW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkSshW
class CK_VISIBLE_PUBLIC CkSshW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkSshW(const CkSshW &);
	CkSshW &operator=(const CkSshW &);

    public:
	CkSshW(void);
	virtual ~CkSshW(void);

	

	static CkSshW *createNew(void);
	

	CkSshW(bool bCallbackOwned);
	static CkSshW *createNew(bool bCallbackOwned);

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	CkBaseProgressW *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgressW *progress);


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

	// Controls whether the caret character '^' is interpreted as indicating a control
	// character. The default value of this property is false. If set to true, then
	// the following sequences are interpreted as control characters in any string
	// passed to SendReqExec or ChannelSendString.
	// ^@ 	00 	00 	NUL  	Null
	// ^A 	01 	01 	SOH  	Start of Heading
	// ^B 	02 	02 	STX  	Start of Text
	// ^C 	03 	03 	ETX  	End of Text
	// ^D 	04 	04 	EOT  	End of Transmission
	// ^E 	05 	05 	ENQ  	Enquiry
	// ^F 	06 	06 	ACK  	Acknowledge
	// ^G 	07 	07 	BEL  	Bell
	// ^H 	08 	08 	BS  	Backspace
	// ^I 	09 	09 	HT  	Character Tabulation, Horizontal Tabulation
	// ^J 	10 	0A 	LF  	Line Feed
	// ^K 	11 	0B 	VT  	Line Tabulation, Vertical Tabulation
	// ^L 	12 	0C 	FF  	Form Feed
	// ^M 	13 	0D 	CR  	Carriage Return
	// ^N 	14 	0E 	SO  	Shift Out
	// ^O 	15 	0F 	SI  	Shift In
	// ^P 	16 	10 	DLE  	Data Link Escape
	// ^Q 	17 	11 	DC1  	Device Control One (XON)
	// ^R 	18 	12 	DC2  	Device Control Two
	// ^S 	19 	13 	DC3  	Device Control Three (XOFF)
	// ^T 	20 	14 	DC4  	Device Control Four
	// ^U 	21 	15 	NAK  	Negative Acknowledge
	// ^V 	22 	16 	SYN  	Synchronous Idle
	// ^W 	23 	17 	ETB  	End of Transmission Block
	// ^X 	24 	18 	CAN  	Cancel
	// ^Y 	25 	19 	EM  	End of medium
	// ^Z 	26 	1A 	SUB  	Substitute
	// ^[ 	27 	1B 	ESC  	Escape
	// ^\ 	28 	1C 	FS  	File Separator
	// ^] 	29 	1D 	GS  	Group Separator
	// ^^ 	30 	1E 	RS  	Record Separator
	// ^_ 	31 	1F 	US  	Unit Separator
	// ^? 	127 	7F 	DEL  	Delete
	bool get_CaretControl(void);
	// Controls whether the caret character '^' is interpreted as indicating a control
	// character. The default value of this property is false. If set to true, then
	// the following sequences are interpreted as control characters in any string
	// passed to SendReqExec or ChannelSendString.
	// ^@ 	00 	00 	NUL  	Null
	// ^A 	01 	01 	SOH  	Start of Heading
	// ^B 	02 	02 	STX  	Start of Text
	// ^C 	03 	03 	ETX  	End of Text
	// ^D 	04 	04 	EOT  	End of Transmission
	// ^E 	05 	05 	ENQ  	Enquiry
	// ^F 	06 	06 	ACK  	Acknowledge
	// ^G 	07 	07 	BEL  	Bell
	// ^H 	08 	08 	BS  	Backspace
	// ^I 	09 	09 	HT  	Character Tabulation, Horizontal Tabulation
	// ^J 	10 	0A 	LF  	Line Feed
	// ^K 	11 	0B 	VT  	Line Tabulation, Vertical Tabulation
	// ^L 	12 	0C 	FF  	Form Feed
	// ^M 	13 	0D 	CR  	Carriage Return
	// ^N 	14 	0E 	SO  	Shift Out
	// ^O 	15 	0F 	SI  	Shift In
	// ^P 	16 	10 	DLE  	Data Link Escape
	// ^Q 	17 	11 	DC1  	Device Control One (XON)
	// ^R 	18 	12 	DC2  	Device Control Two
	// ^S 	19 	13 	DC3  	Device Control Three (XOFF)
	// ^T 	20 	14 	DC4  	Device Control Four
	// ^U 	21 	15 	NAK  	Negative Acknowledge
	// ^V 	22 	16 	SYN  	Synchronous Idle
	// ^W 	23 	17 	ETB  	End of Transmission Block
	// ^X 	24 	18 	CAN  	Cancel
	// ^Y 	25 	19 	EM  	End of medium
	// ^Z 	26 	1A 	SUB  	Substitute
	// ^[ 	27 	1B 	ESC  	Escape
	// ^\ 	28 	1C 	FS  	File Separator
	// ^] 	29 	1D 	GS  	Group Separator
	// ^^ 	30 	1E 	RS  	Record Separator
	// ^_ 	31 	1F 	US  	Unit Separator
	// ^? 	127 	7F 	DEL  	Delete
	void put_CaretControl(bool newVal);

	// If a request to open a channel fails, this property contains a code that
	// identifies the reason for failure. The reason codes are defined in RFC 4254 and
	// are reproduced here:
	//              Symbolic name                           reason code
	//              -------------                           -----------
	//             SSH_OPEN_ADMINISTRATIVELY_PROHIBITED          1
	//             SSH_OPEN_CONNECT_FAILED                       2
	//             SSH_OPEN_UNKNOWN_CHANNEL_TYPE                 3
	//             SSH_OPEN_RESOURCE_SHORTAGE                    4
	int get_ChannelOpenFailCode(void);

	// The descriptive text corresponding to the ChannelOpenFailCode property.
	void get_ChannelOpenFailReason(CkString &str);
	// The descriptive text corresponding to the ChannelOpenFailCode property.
	const wchar_t *channelOpenFailReason(void);

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

	// Maximum number of milliseconds to wait when connecting to an SSH server.
	int get_ConnectTimeoutMs(void);
	// Maximum number of milliseconds to wait when connecting to an SSH server.
	void put_ConnectTimeoutMs(int newVal);

	// If the SSH server sent a DISCONNECT message when closing the connection, this
	// property contains the "reason code" as specified in RFC 4253:
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

	// If the SSH/ server sent a DISCONNECT message when closing the connection, this
	// property contains a descriptive string for the "reason code" as specified in RFC
	// 4253.
	void get_DisconnectReason(CkString &str);
	// If the SSH/ server sent a DISCONNECT message when closing the connection, this
	// property contains a descriptive string for the "reason code" as specified in RFC
	// 4253.
	const wchar_t *disconnectReason(void);

	// Enables or disables the use of compression w/ the SSH connection. The default
	// value is true, meaning that compression is used if the server supports it.
	// 
	// Some older SSH servers have been found that claim to support compression, but
	// actually fail when compression is used. PuTTY does not enable compression by
	// default. If trouble is encountered where the SSH server disconnects immediately
	// after the connection is seemingly established (i.e. during authentication), then
	// check to see if disabling compression resolves the problem.
	// 
	bool get_EnableCompression(void);
	// Enables or disables the use of compression w/ the SSH connection. The default
	// value is true, meaning that compression is used if the server supports it.
	// 
	// Some older SSH servers have been found that claim to support compression, but
	// actually fail when compression is used. PuTTY does not enable compression by
	// default. If trouble is encountered where the SSH server disconnects immediately
	// after the connection is seemingly established (i.e. during authentication), then
	// check to see if disabling compression resolves the problem.
	// 
	void put_EnableCompression(bool newVal);

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

	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any SSH operation prior to
	// completion. If HeartbeatMs is 0 (the default), no AbortCheck event callbacks
	// will fire.
	int get_HeartbeatMs(void);
	// This is the number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort any SSH operation prior to
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

	// Set after connecting to an SSH server. The format of the fingerprint looks like
	// this: "ssh-rsa 1024 68:ff:d1:4e:6c:ff:d7:b0:d6:58:73:85:07:bc:2e:d5"
	void get_HostKeyFingerprint(CkString &str);
	// Set after connecting to an SSH server. The format of the fingerprint looks like
	// this: "ssh-rsa 1024 68:ff:d1:4e:6c:ff:d7:b0:d6:58:73:85:07:bc:2e:d5"
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

	// Causes SSH operations to fail when progress for sending or receiving data halts
	// for more than this number of milliseconds. Setting IdleTimeoutMs = 0 (the
	// default) allows the application to wait indefinitely.
	int get_IdleTimeoutMs(void);
	// Causes SSH operations to fail when progress for sending or receiving data halts
	// for more than this number of milliseconds. Setting IdleTimeoutMs = 0 (the
	// default) allows the application to wait indefinitely.
	void put_IdleTimeoutMs(int newVal);

	// Returns true if the component is connected to an SSH server.
	// 
	// Note: The IsConnected property is set to true after successfully completing
	// the Connect method call. The IsConnected property will only be set to false by
	// calling Disconnect, or by the failure of another method call such that the
	// disconnection is detected.
	// 
	bool get_IsConnected(void);

	// Controls whether communications to/from the SSH server are saved to the
	// SessionLog property. The default value is false. If this property is set to
	// true, the contents of the SessionLog property will continuously grow as SSH
	// activity transpires. The purpose of the KeepSessionLog / SessionLog properties
	// is to help in debugging any future problems that may arise.
	bool get_KeepSessionLog(void);
	// Controls whether communications to/from the SSH server are saved to the
	// SessionLog property. The default value is false. If this property is set to
	// true, the contents of the SessionLog property will continuously grow as SSH
	// activity transpires. The purpose of the KeepSessionLog / SessionLog properties
	// is to help in debugging any future problems that may arise.
	void put_KeepSessionLog(bool newVal);

	// The maximum packet length to be used in the SSH transport protocol. The default
	// value is 8192.
	// 
	// Note: If a large amount of data is to be flowing to/from the SSH server, then
	// setting the MaxPacketSize equal to 32768 may improve performance. For those
	// familiar with the inner workings of the SSH protocol, this is the "maximum
	// packet size" value that is sent in the SSH_MSG_CHANNEL_OPEN message as defined
	// in RFC 4254.
	// 
	int get_MaxPacketSize(void);
	// The maximum packet length to be used in the SSH transport protocol. The default
	// value is 8192.
	// 
	// Note: If a large amount of data is to be flowing to/from the SSH server, then
	// setting the MaxPacketSize equal to 32768 may improve performance. For those
	// familiar with the inner workings of the SSH protocol, this is the "maximum
	// packet size" value that is sent in the SSH_MSG_CHANNEL_OPEN message as defined
	// in RFC 4254.
	// 
	void put_MaxPacketSize(int newVal);

	// The number of currently open channels.
	int get_NumOpenChannels(void);

	// Set by the AuthenticatePw and AuthenticatePwPk methods. If the authenticate
	// method returns a failed status, and this property is set to true, then it
	// indicates the server requested a password change. In this case, re-call the
	// authenticate method, but provide both the old and new passwords in the following
	// format, where vertical bar characters encapsulate the old and new passwords:
	// 
	//     |oldPassword|newPassword|
	// 
	bool get_PasswordChangeRequested(void);

	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	bool get_PreferIpv6(void);
	// If true, then use IPv6 over IPv4 when both are supported for a particular
	// domain. The default value of this property is false, which will choose IPv4
	// over IPv6.
	void put_PreferIpv6(bool newVal);

	// The maximum amount of time to allow for reading messages/data from the SSH
	// server. This is different from the IdleTimeoutMs property. The IdleTimeoutMs is
	// the maximum amount of time to wait while no incoming data is arriving. The
	// ReadTimeoutMs is the maximum amount of time to allow for reading data even if
	// data is continuing to arrive. The default value of 0 indicates an infinite
	// timeout value.
	int get_ReadTimeoutMs(void);
	// The maximum amount of time to allow for reading messages/data from the SSH
	// server. This is different from the IdleTimeoutMs property. The IdleTimeoutMs is
	// the maximum amount of time to wait while no incoming data is arriving. The
	// ReadTimeoutMs is the maximum amount of time to allow for reading data even if
	// data is continuing to arrive. The default value of 0 indicates an infinite
	// timeout value.
	void put_ReadTimeoutMs(int newVal);

	// Indicates the charset to be used for the command sent via the SendReqExec
	// method. The default is "ANSI". A likely alternate value would be "utf-8".
	void get_ReqExecCharset(CkString &str);
	// Indicates the charset to be used for the command sent via the SendReqExec
	// method. The default is "ANSI". A likely alternate value would be "utf-8".
	const wchar_t *reqExecCharset(void);
	// Indicates the charset to be used for the command sent via the SendReqExec
	// method. The default is "ANSI". A likely alternate value would be "utf-8".
	void put_ReqExecCharset(const wchar_t *newVal);

	// The server-identifier string received from the server during connection
	// establishment. For example, a typical value would be similar to
	// "SSH-2.0-OpenSSH_7.2p2 Ubuntu-4ubuntu2.2".
	void get_ServerIdentifier(CkString &str);
	// The server-identifier string received from the server during connection
	// establishment. For example, a typical value would be similar to
	// "SSH-2.0-OpenSSH_7.2p2 Ubuntu-4ubuntu2.2".
	const wchar_t *serverIdentifier(void);

	// Contains a log of the messages sent to/from the SSH server. To enable session
	// logging, set the KeepSessionLog property = true. Note: This property is not a
	// filename -- it is a string property that contains the session log data.
	void get_SessionLog(CkString &str);
	// Contains a log of the messages sent to/from the SSH server. To enable session
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

	// If true, then stderr is redirected to stdout. In this case, channel output for
	// both stdout and stderr is combined and retrievable via the following methods:
	// GetReceivedData, GetReceivedDataN, GetReceivedText, GetReceivedTextS. If this
	// property is false, then stderr is available separately via the
	// GetReceivedStderr method.
	// 
	// The default value of this property is true.
	// 
	// Note: Most SSH servers do not send stderr output as "extended data" packets as
	// specified in RFC 4254. The SessionLog may be examined to see if any
	// CHANNEL_EXTENDED_DATA messages exist. If not, then all of the output (stdout +
	// stderr) was sent via CHANNEL_DATA messages, and therefore it is not possible to
	// differentiate stderr output from stdout. In summary: This feature will not work
	// for most SSH servers.
	// 
	bool get_StderrToStdout(void);
	// If true, then stderr is redirected to stdout. In this case, channel output for
	// both stdout and stderr is combined and retrievable via the following methods:
	// GetReceivedData, GetReceivedDataN, GetReceivedText, GetReceivedTextS. If this
	// property is false, then stderr is available separately via the
	// GetReceivedStderr method.
	// 
	// The default value of this property is true.
	// 
	// Note: Most SSH servers do not send stderr output as "extended data" packets as
	// specified in RFC 4254. The SessionLog may be examined to see if any
	// CHANNEL_EXTENDED_DATA messages exist. If not, then all of the output (stdout +
	// stderr) was sent via CHANNEL_DATA messages, and therefore it is not possible to
	// differentiate stderr output from stdout. In summary: This feature will not work
	// for most SSH servers.
	// 
	void put_StderrToStdout(bool newVal);

	// If true, then terminal color codes are stripped from the received text. The
	// default value of this property is true. (Color codes are non-printable escape
	// sequences that provide information about color for text in a terminal.)
	bool get_StripColorCodes(void);
	// If true, then terminal color codes are stripped from the received text. The
	// default value of this property is true. (Color codes are non-printable escape
	// sequences that provide information about color for text in a terminal.)
	void put_StripColorCodes(bool newVal);

	// Controls whether the TCP_NODELAY socket option is used for the underlying TCP/IP
	// socket. The default value is true. This disables the Nagle algorithm and
	// allows for better performance when small amounts of data are sent to/from the
	// SSH server.
	bool get_TcpNoDelay(void);
	// Controls whether the TCP_NODELAY socket option is used for the underlying TCP/IP
	// socket. The default value is true. This disables the Nagle algorithm and
	// allows for better performance when small amounts of data are sent to/from the
	// SSH server.
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
	// 
	void put_UncommonOptions(const wchar_t *newVal);

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



	// ----------------------
	// Methods
	// ----------------------
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
	// An SSH session always begins by first calling Connect to connect to the SSH
	// server, and then calling either AuthenticatePw or AuthenticatePk to login.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
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

	// The same as AuthenticatePw, except the login and passwords strings are passed in
	// secure string objects.
	bool AuthenticateSecPw(CkSecureStringW &login, CkSecureStringW &password);

	// Creates an asynchronous task to call the AuthenticateSecPw method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *AuthenticateSecPwAsync(CkSecureStringW &login, CkSecureStringW &password);

	// The same as AuthenticatePwPk, except the login and passwords strings are passed
	// in secure string objects.
	bool AuthenticateSecPwPk(CkSecureStringW &username, CkSecureStringW &password, CkSshKeyW &privateKey);

	// Creates an asynchronous task to call the AuthenticateSecPwPk method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *AuthenticateSecPwPkAsync(CkSecureStringW &username, CkSecureStringW &password, CkSshKeyW &privateKey);

	// Returns true if the channel indicated by channelNum is open. Otherwise returns
	// false.
	bool ChannelIsOpen(int channelNum);

	// Polls for incoming data on an open channel. This method will read a channel,
	// waiting at most pollTimeoutMs milliseconds for data to arrive. Return values are as
	// follows:
	// 
	// -1 -- Error. Check the IsConnected property to see if the connection to the SSH
	// server is still valid. Also, call ChannelIsOpen to see if the channel remains
	// open. The LastErrorText property will contain more detailed information
	// regarding the error.
	// 
	// -2 -- No additional data was received prior to the poll timeout.
	// 
	// >0 -- Additional data was received and the return value indicates how many bytes
	// are available to be "picked up". Methods that read data on a channel do not
	// return the received data directly. Instead, they return an integer to indicate
	// how many bytes are available to be "picked up". An application picks up the
	// available data by calling GetReceivedData or GetReceivedText.
	// 
	// =0 -- A zero can be returned if the channel EOF has already been received, or if
	// the channel had already been closed.
	// 
	int ChannelPoll(int channelNum, int pollTimeoutMs);

	// Creates an asynchronous task to call the ChannelPoll method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelPollAsync(int channelNum, int pollTimeoutMs);

	// Reads incoming data on an open channel. This method will read a channel, waiting
	// at most IdleTimeoutMs milliseconds for data to arrive. Return values are as
	// follows:
	// 
	// -1 -- Error. Check the IsConnected property to see if the connection to the SSH
	// server is still valid. Also, call ChannelIsOpen to see if the channel remains
	// open. The LastErrorText property will contain more detailed information
	// regarding the error.
	// 
	// -2 -- No additional data was received prior to the IdleTimeoutMs timeout.
	// 
	// >0 -- Additional data was received and the return value indicates how many bytes
	// are available to be "picked up". Methods that read data on a channel do not
	// return the received data directly. Instead, they return an integer to indicate
	// how many bytes are available to be "picked up". An application picks up the
	// available data by calling GetReceivedData or GetReceivedText.
	// 
	// =0 -- A zero can be returned if the channel EOF has already been received, or if
	// the channel had already been closed.
	// 
	int ChannelRead(int channelNum);

	// Creates an asynchronous task to call the ChannelRead method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelReadAsync(int channelNum);

	// Reads incoming data on an open channel and continues reading until no data
	// arrives for pollTimeoutMs milliseconds. The first read will wait a max of IdleTimeoutMs
	// milliseconds before timing out. Subsequent reads wait a max of pollTimeoutMs milliseconds
	// before timing out.
	// 
	// The idea behind ChannelReadAndPoll is to capture the output of a shell command.
	// One might imagine the typical sequence of events when executing a shell command
	// to be like this: (1) client sends command to server, (2) server executes the
	// command (i.e. it's computing...), potentially taking some amount of time, (3)
	// output is streamed back to the client. It makes sense for the client to wait a
	// longer period of time for the first data to arrive, but once it begins arriving,
	// the timeout can be shortened. This is exactly what ChannelReadAndPoll does --
	// the first timeout is controlled by the IdleTimeoutMs property, while the
	// subsequent reads (once output starts flowing) is controlled by pollTimeoutMs.
	// 
	// Return values are as follows:
	// -1 -- Error. Check the IsConnected property to see if the connection to the SSH
	// server is still valid. Also, call ChannelIsOpen to see if the channel remains
	// open. The LastErrorText property will contain more detailed information
	// regarding the error.
	// 
	// -2 -- No additional data was received prior to the IdleTimeoutMs timeout.
	// 
	// >0 -- Additional data was received and the return value indicates how many bytes
	// are available to be "picked up". Methods that read data on a channel do not
	// return the received data directly. Instead, they return an integer to indicate
	// how many bytes are available to be "picked up". An application picks up the
	// available data by calling GetReceivedData or GetReceivedText.
	// 
	// =0 -- A zero can be returned if the channel EOF has already been received, or if
	// the channel had already been closed.
	// 
	int ChannelReadAndPoll(int channelNum, int pollTimeoutMs);

	// Creates an asynchronous task to call the ChannelReadAndPoll method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelReadAndPollAsync(int channelNum, int pollTimeoutMs);

	// The same as ChannelReadAndPoll, except this method will return as soon as maxNumBytes
	// is exceeded.
	int ChannelReadAndPoll2(int channelNum, int pollTimeoutMs, int maxNumBytes);

	// Creates an asynchronous task to call the ChannelReadAndPoll2 method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelReadAndPoll2Async(int channelNum, int pollTimeoutMs, int maxNumBytes);

	// true if a CLOSE message has been received on the channel indicated by channelNum.
	// When a CLOSE is received, no subsequent data should be sent in either direction
	// -- the channel is closed in both directions.
	bool ChannelReceivedClose(int channelNum);

	// true if an EOF message has been received on the channel indicated by channelNum.
	// When an EOF is received, no more data will be forthcoming on the channel.
	// However, data may still be sent in the opposite direction.
	bool ChannelReceivedEof(int channelNum);

	// true if an exit status code was received on the channel. Otherwise false.
	bool ChannelReceivedExitStatus(int channelNum);

	// Reads incoming data on an open channel until the channel is closed by the
	// server. If successful, the number of bytes available to be "picked up" can be
	// determined by calling GetReceivedNumBytes. The received data may be retrieved by
	// calling GetReceivedData or GetReceivedText.
	bool ChannelReceiveToClose(int channelNum);

	// Creates an asynchronous task to call the ChannelReceiveToClose method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelReceiveToCloseAsync(int channelNum);

	// Reads incoming text data on an open channel until the received data matches the
	// matchPattern. For example, to receive data until the string "Hello World" arrives, set
	// matchPattern equal to "*Hello World*". charset indicates the character encoding of the text
	// being received ("iso-8859-1" for example). caseSensitive may be set to true for case
	// sensitive matching, or false for case insensitive matching.
	// 
	// Returns true if text data matching matchPattern was received and is available to be
	// picked up by calling GetReceivedText (or GetReceivedTextS). IMPORTANT: This
	// method may read beyond the matching text. Call GetReceivedTextS to extract only
	// the data up-to and including the matching text.
	// 
	// Important Notes:
	//     It's wise to set the ReadTimeoutMs property to a non-zero value to prevent
	//     an infinite wait if the matchPattern never arrives.
	//     If using a shell session and SendReqPty was called, set the termType =
	//     "dumb". If terminal control codes get mixed into the output stream, it could
	//     disrupt matching.
	//     Be aware of the StderrToStdout property setting. The default value is true,
	//     which means that stderr is mixed with stdout in the output stream. This could
	//     disrupt matching. Set StderrToStdout to false to prevent this possibility.
	// 
	bool ChannelReceiveUntilMatch(int channelNum, const wchar_t *matchPattern, const wchar_t *charset, bool caseSensitive);

	// Creates an asynchronous task to call the ChannelReceiveUntilMatch method with
	// the arguments provided. (Async methods are available starting in Chilkat
	// v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelReceiveUntilMatchAsync(int channelNum, const wchar_t *matchPattern, const wchar_t *charset, bool caseSensitive);

	// The same as ChannelReceiveUntilMatch except that the method returns when any one
	// of the match patterns specified in matchPatterns are received on the channel.
	// 
	// Important: It's wise to set the ReadTimeoutMs property to a non-zero value to
	// prevent an infinite wait if of the matchPatterns ever arrives.
	// 
	bool ChannelReceiveUntilMatchN(int channelNum, CkStringArrayW &matchPatterns, const wchar_t *charset, bool caseSensitive);

	// Creates an asynchronous task to call the ChannelReceiveUntilMatchN method with
	// the arguments provided. (Async methods are available starting in Chilkat
	// v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelReceiveUntilMatchNAsync(int channelNum, CkStringArrayW &matchPatterns, const wchar_t *charset, bool caseSensitive);

	// Releases the internal memory resources for a channel previously opened by
	// OpenSessionChannel, OpenCustomChannel, or OpenDirectTcpIpChannel. It is not
	// absolutely necessary to call this method because the internal memory resources
	// for all channels are automatically released when the SSH object instance is
	// deleted/disposed. This method becomes necessary only when an extremely large
	// number of channels within the same SSH object instance are opened, used, and
	// closed over a long period of time.
	void ChannelRelease(int channelNum);

	// Sends a CLOSE message to the server for the channel indicated by channelNum. This
	// closes both directions of the bidirectional channel.
	bool ChannelSendClose(int channelNum);

	// Creates an asynchronous task to call the ChannelSendClose method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelSendCloseAsync(int channelNum);

	// Sends byte data on the channel indicated by channelNum.
	bool ChannelSendData(int channelNum, CkByteData &byteData);

	// Creates an asynchronous task to call the ChannelSendData method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelSendDataAsync(int channelNum, CkByteData &byteData);

	// Sends an EOF for the channel indicated by channelNum. Once an EOF is sent, no
	// additional data may be sent on the channel. However, the channel remains open
	// and additional data may still be received from the server.
	bool ChannelSendEof(int channelNum);

	// Creates an asynchronous task to call the ChannelSendEof method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelSendEofAsync(int channelNum);

	// Sends character data on the channel indicated by channelNum. The text is converted to
	// the charset indicated by charset prior to being sent. A list of supported charset
	// values may be found on this page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	bool ChannelSendString(int channelNum, const wchar_t *textData, const wchar_t *charset);

	// Creates an asynchronous task to call the ChannelSendString method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ChannelSendStringAsync(int channelNum, const wchar_t *textData, const wchar_t *charset);

	// Returns true if the underlying TCP socket is connected to the SSH server.
	bool CheckConnection(void);

	// Clears the collection of TTY modes that are sent with the SendReqPty method.
	void ClearTtyModes(void);

	// Connects to the SSH server at domainName:port
	// 
	// The domainName may be a domain name or an IPv4 or IPv6 address in string format.
	// 
	// Internally, the following SSH connection protocol algorithms are supported:
	//     Hostkey Types: ssh-rsa, ssh-dsa, ecdsa-sha2-nistp256, rsa-sha2-256,
	//     rsa-sha2-512, ssh-ed25519
	//     Key Exchange Methods: curve25519-sha256@libssh.org, ecdh-sha2-nistp256,
	//     ecdh-sha2-nistp384, ecdh-sha2-nistp521, diffie-hellman-group-exchange-sha256,
	//     diffie-hellman-group-exchange-sha1, diffie-hellman-group14-sha1,
	//     diffie-hellman-group1-sha1
	//     Ciphers: chacha20-poly1305@openssh.com, aes256-ctr, aes192-ctr, aes128-ctr,
	//     aes256-cbc, aes192-cbc, aes128-cbc, twofish256-cbc, twofish128-cbc,
	//     blowfish-cbc, 3des-cbc, arcfour128, arcfour256
	//     MAC Algorithms: hmac-sha2-256, hmac-sha2-512, hmac-sha1, hmac-md5,
	//     hmac-ripemd160, hmac-sha1-96
	//     Compression: none, zlib, zlib@openssh.com
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

	// Connects to an SSH server through an existing SSH connection. The ssh is an
	// existing connected and authenticated SSH object. The connection to hostname:port is
	// made through the existing SSH connection via port-forwarding. If successful, the
	// connection is as follows: application => ServerSSH1 => ServerSSH2. (where
	// ServerSSH1 is the ssh and ServerSSH2 is the SSH server at hostname:port) Once
	// connected in this way, all communications are routed through ServerSSH1 to
	// ServerSSH2. This includes authentication -- which means the application must
	// still call one of the Authenticate* methods to authenticate with ServerSSH2.
	bool ConnectThroughSsh(CkSshW &ssh, const wchar_t *hostname, int port);

	// Creates an asynchronous task to call the ConnectThroughSsh method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ConnectThroughSshAsync(CkSshW &ssh, const wchar_t *hostname, int port);

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

	// Disconnects from the SSH server.
	void Disconnect(void);

	// Queries the SSH server to find out which authentication methods it supports.
	// Returns a string such as "publickey,password,keyboard-interactive".
	// 
	// This method should be called after connecting, but prior to authenticating. The
	// method intentionally disconnects from the server after getting the
	// authentication methods. An application may then connect again and authentication
	// with a chosen method. (In most cases, an application knows in advance the type
	// of authentication to be used, and this method is never called.)
	// 
	bool GetAuthMethods(CkString &outStr);
	// Queries the SSH server to find out which authentication methods it supports.
	// Returns a string such as "publickey,password,keyboard-interactive".
	// 
	// This method should be called after connecting, but prior to authenticating. The
	// method intentionally disconnects from the server after getting the
	// authentication methods. An application may then connect again and authentication
	// with a chosen method. (In most cases, an application knows in advance the type
	// of authentication to be used, and this method is never called.)
	// 
	const wchar_t *getAuthMethods(void);
	// Queries the SSH server to find out which authentication methods it supports.
	// Returns a string such as "publickey,password,keyboard-interactive".
	// 
	// This method should be called after connecting, but prior to authenticating. The
	// method intentionally disconnects from the server after getting the
	// authentication methods. An application may then connect again and authentication
	// with a chosen method. (In most cases, an application knows in advance the type
	// of authentication to be used, and this method is never called.)
	// 
	const wchar_t *authMethods(void);

	// Creates an asynchronous task to call the GetAuthMethods method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetAuthMethodsAsync(void);

	// Returns the exit status code for a channel. This method should only be called if
	// an exit status has been received. You may check to see if the exit status was
	// received by calling ChannelReceivedExitStatus.
	int GetChannelExitStatus(int channelNum);

	// Returns the channel number for the Nth open channel. Indexing begins at 0, and
	// the number of currently open channels is indicated by the NumOpenChannels
	// property. Returns -1 if the index is out of range.
	int GetChannelNumber(int index);

	// Returns a string describing the channel type for the Nth open channel. Channel
	// types are: "session", "x11", "forwarded-tcpip", and "direct-tcpip".
	bool GetChannelType(int index, CkString &outStr);
	// Returns a string describing the channel type for the Nth open channel. Channel
	// types are: "session", "x11", "forwarded-tcpip", and "direct-tcpip".
	const wchar_t *getChannelType(int index);
	// Returns a string describing the channel type for the Nth open channel. Channel
	// types are: "session", "x11", "forwarded-tcpip", and "direct-tcpip".
	const wchar_t *channelType(int index);

	// Returns the accumulated data received on the channel indicated by channelNum and
	// clears the channel's internal receive buffer.
	bool GetReceivedData(int channelNum, CkByteData &outBytes);

	// Same as GetReceivedData, but a maximum of maxNumBytes bytes is returned.
	bool GetReceivedDataN(int channelNum, int maxNumBytes, CkByteData &outBytes);

	// Returns the number of bytes available in the internal receive buffer for the
	// specified channelNum. The received data may be retrieved by calling GetReceivedData or
	// GetReceivedText.
	int GetReceivedNumBytes(int channelNum);

	// Returns the accumulated stderr bytes received on the channel indicated by channelNum
	// and clears the channel's internal stderr receive buffer.
	// 
	// Note: If the StderrToStdout property is set to true, then stderr is
	// automatically redirected to stdout. This is the default behavior. The following
	// methods can be called to retrieve the channel's stdout: GetReceivedData,
	// GetReceivedDataN, GetReceivedText, and GetReceivedTextS.
	// 
	bool GetReceivedStderr(int channelNum, CkByteData &outBytes);

	// Returns the accumulated stderr text received on the channel indicated by channelNum
	// and clears the channel's internal receive buffer. The charset indicates the charset
	// of the character data in the internal receive buffer. A list of supported
	// charset values may be found on this page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	bool GetReceivedStderrText(int channelNum, const wchar_t *charset, CkString &outStr);
	// Returns the accumulated stderr text received on the channel indicated by channelNum
	// and clears the channel's internal receive buffer. The charset indicates the charset
	// of the character data in the internal receive buffer. A list of supported
	// charset values may be found on this page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	const wchar_t *getReceivedStderrText(int channelNum, const wchar_t *charset);
	// Returns the accumulated stderr text received on the channel indicated by channelNum
	// and clears the channel's internal receive buffer. The charset indicates the charset
	// of the character data in the internal receive buffer. A list of supported
	// charset values may be found on this page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	const wchar_t *receivedStderrText(int channelNum, const wchar_t *charset);

	// Returns the accumulated text received on the channel indicated by channelNum and
	// clears the channel's internal receive buffer. The charset indicates the charset of
	// the character data in the internal receive buffer. A list of supported charset
	// values may be found on this page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	bool GetReceivedText(int channelNum, const wchar_t *charset, CkString &outStr);
	// Returns the accumulated text received on the channel indicated by channelNum and
	// clears the channel's internal receive buffer. The charset indicates the charset of
	// the character data in the internal receive buffer. A list of supported charset
	// values may be found on this page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	const wchar_t *getReceivedText(int channelNum, const wchar_t *charset);
	// Returns the accumulated text received on the channel indicated by channelNum and
	// clears the channel's internal receive buffer. The charset indicates the charset of
	// the character data in the internal receive buffer. A list of supported charset
	// values may be found on this page:Supported Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	const wchar_t *receivedText(int channelNum, const wchar_t *charset);

	// The same as GetReceivedText, except only the text up to and including substr is
	// returned. The text returned is removed from the internal receive buffer. If the
	// substr was not found in the internal receive buffer, an empty string is returned
	// and the internal receive buffer is not modified.
	bool GetReceivedTextS(int channelNum, const wchar_t *substr, const wchar_t *charset, CkString &outStr);
	// The same as GetReceivedText, except only the text up to and including substr is
	// returned. The text returned is removed from the internal receive buffer. If the
	// substr was not found in the internal receive buffer, an empty string is returned
	// and the internal receive buffer is not modified.
	const wchar_t *getReceivedTextS(int channelNum, const wchar_t *substr, const wchar_t *charset);
	// The same as GetReceivedText, except only the text up to and including substr is
	// returned. The text returned is removed from the internal receive buffer. If the
	// substr was not found in the internal receive buffer, an empty string is returned
	// and the internal receive buffer is not modified.
	const wchar_t *receivedTextS(int channelNum, const wchar_t *substr, const wchar_t *charset);

	// Provides information about what transpired in the last method called. For many
	// methods, there is no information. For some methods, details about what
	// transpired can be obtained via LastJsonData.
	// The caller is responsible for deleting the object returned by this method.
	CkJsonObjectW *LastJsonData(void);

	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Opens a custom channel with a custom server that uses the SSH protocol. The channelType
	// is application-defined.
	// 
	// If successful, the channel number is returned. This is the number that should be
	// passed to any method requiring a channel number. A -1 is returned upon failure.
	// 
	int OpenCustomChannel(const wchar_t *channelType);

	// Creates an asynchronous task to call the OpenCustomChannel method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpenCustomChannelAsync(const wchar_t *channelType);

	// Open a direct-tcpip channel for port forwarding. Data sent on the channel via
	// ChannelSend* methods is sent to the SSH server and then forwarded to targetHostname:targetPort.
	// The SSH server automatically forwards data received from targetHostname:targetPort to the SSH
	// client. Therefore, calling ChannelRead* and ChannelReceive* methods is
	// equivalent to reading directly from targetHostname:targetPort.
	// 
	// If successful, the channel number is returned. This is the number that should be
	// passed to any method requiring a channel number. A -1 is returned upon failure.
	// 
	int OpenDirectTcpIpChannel(const wchar_t *targetHostname, int targetPort);

	// Creates an asynchronous task to call the OpenDirectTcpIpChannel method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpenDirectTcpIpChannelAsync(const wchar_t *targetHostname, int targetPort);

	// Opens a new session channel. Almost everything you will do with the Chilkat SSH
	// component will involve opening a session channel. The normal sequence of
	// operation is typically this: 1) Connect to the SSH server. 2) Authenticate. 3)
	// Open a session channel. 4) do something on the channel such as opening a shell,
	// execute a command, etc.
	// 
	// If successful, the channel number is returned. This is the number that should be
	// passed to any method requiring a channel number. A -1 is returned upon failure.
	// 
	int OpenSessionChannel(void);

	// Creates an asynchronous task to call the OpenSessionChannel method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *OpenSessionChannelAsync(void);

	// This is the same as GetReceivedText, except the internal receive buffer is not
	// cleared.
	bool PeekReceivedText(int channelNum, const wchar_t *charset, CkString &outStr);
	// This is the same as GetReceivedText, except the internal receive buffer is not
	// cleared.
	const wchar_t *peekReceivedText(int channelNum, const wchar_t *charset);

	// Returns a channel number for a completed command that was previously sent via
	// QuickCmdSend. Returns -1 if no commands have yet completed. The pollTimeoutMs indicates
	// how long to wait (in milliseconds) for any command in progress (on any channel)
	// to complete before returning -1.
	// 
	// Returns -2 if an error occurred (for example, if the connection to the SSH
	// server was lost while checking for responses).
	// 
	int QuickCmdCheck(int pollTimeoutMs);

	// Creates an asynchronous task to call the QuickCmdCheck method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *QuickCmdCheckAsync(int pollTimeoutMs);

	// Sends a command and returns the channel number for the command that has started.
	// This is the equivalent of calling OpenSessionChannel, followed by SendReqExec. A
	// value of -1 is returned on failure.
	// 
	// The ReqExecCharset property controls the charset used for the command that is
	// sent.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	int QuickCmdSend(const wchar_t *command);

	// Creates an asynchronous task to call the QuickCmdSend method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *QuickCmdSendAsync(const wchar_t *command);

	// Simplified method for executing a remote command and getting the complete
	// output. This is the equivalent of calling OpenSessionChannel, followed by
	// SendReqExec, then ChannelReceiveToClose, and finally GetReceivedText.
	// 
	// The charset indicates the charset of the command's output (such as "utf-8" or
	// "ansi"). A list of supported charset values may be found on this page:Supported
	// Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// The ReqExecCharset property controls the charset used for the command that is
	// sent.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	bool QuickCommand(const wchar_t *command, const wchar_t *charset, CkString &outStr);
	// Simplified method for executing a remote command and getting the complete
	// output. This is the equivalent of calling OpenSessionChannel, followed by
	// SendReqExec, then ChannelReceiveToClose, and finally GetReceivedText.
	// 
	// The charset indicates the charset of the command's output (such as "utf-8" or
	// "ansi"). A list of supported charset values may be found on this page:Supported
	// Charsets
	// <http://www.chilkatsoft.com/p/p_463.asp>.
	// 
	// The ReqExecCharset property controls the charset used for the command that is
	// sent.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	const wchar_t *quickCommand(const wchar_t *command, const wchar_t *charset);

	// Creates an asynchronous task to call the QuickCommand method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *QuickCommandAsync(const wchar_t *command, const wchar_t *charset);

	// Simplified method for starting a remote shell session. It is the equivalent of
	// calling OpenSessionChannel, followed by SendReqPty, and finally SendReqShell.
	// 
	// Returns the SSH channel number for the session, or -1 if not successful.
	// 
	// Important: When reporting problems, please send the full contents of the
	// LastErrorText property to support@chilkatsoft.com.
	// 
	int QuickShell(void);

	// Creates an asynchronous task to call the QuickShell method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *QuickShellAsync(void);

	// Initiates a re-key with the SSH server. The ReKey method does not return until
	// the key re-exchange is complete.
	// 
	// RFC 4253 (the SSH Transport Layer Protocol) recommends that keys be changed
	// after each gigabyte of transmitted data or after each hour of connection time,
	// whichever comes sooner. Key re-exchange is a public-key operation and requires a
	// fair amount of processing power and should not be performed too often. Either
	// side (client or server) may initiate a key re-exchange at any time.
	// 
	// In most cases, a server will automatically initiate key re-exchange whenever it
	// deems necessary, and the Chilkat SSH component handles these transparently. For
	// example, if the Chilkat SSH component receives a re-key message from the server
	// while in the process of receiving data on a channel, it will automatically
	// handle the key re-exchange and the application will not even realize that an
	// underlying key re-exchange occurred.
	// 
	bool ReKey(void);

	// Creates an asynchronous task to call the ReKey method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReKeyAsync(void);

	// Sends an IGNORE message to the SSH server. This is one way of verifying that the
	// connection to the SSH server is open and valid. The SSH server does not respond
	// to an IGNORE message. It simply ignores it. IGNORE messages are not associated
	// with a channel (in other words, you do not need to first open a channel prior to
	// sending an IGNORE message).
	bool SendIgnore(void);

	// Creates an asynchronous task to call the SendIgnore method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendIgnoreAsync(void);

	// Initiates execution of a command on the channel specified by channelNum. The commandLine
	// contains the full command line including any command-line parameters (just as
	// you would type the command at a shell prompt).
	// 
	// This is the equivalent of running a command on a remote server via the "rexec"
	// command. Seerexec command
	// <https://linux.die.net/man/1/rexec>. It is not a shell session.
	// 
	// Important: A channel only exists for a single request. You may not call
	// SendReqExec multiple times on the same open channel. The reason is that the SSH
	// server automatically closes the channel at the end of the exec. The solution is
	// to call OpenSessionChannel to get a new channel, and then call SendReqExec using
	// the new channel. It is OK to have more than one channel open simultaneously.
	// 
	// Charset: The ReqExecCharset property has been added in version 9.5.0.47. This
	// can be set to control the character encoding of the command sent to the server.
	// The default is ANSI. A likely alternative value is "utf-8".
	// 
	bool SendReqExec(int channelNum, const wchar_t *commandLine);

	// Creates an asynchronous task to call the SendReqExec method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqExecAsync(int channelNum, const wchar_t *commandLine);

	// Requests a pseudo-terminal for a session channel. If the termType is a character
	// oriented terminal ("vt100" for example), then widthInChars and heightInChars would be set to
	// non-zero values, while widthInPixels and heightInPixels may be set to 0. If termType is pixel-oriented,
	// such as "xterm", the reverse is true (i.e. set widthInPixels and heightInPixels, but set widthInChars and
	// heightInChars equal to 0).
	// 
	// In most cases, you probably don't even want terminal emulation. In that case,
	// try setting termType = "dumb". Terminal emulation causes terminal escape sequences
	// to be included with shell command output. A "dumb" terminal should have no
	// escape sequences.
	// 
	// Some SSH servers allow a shell to be started (via the SendReqShell method)
	// without the need to first request a pseudo-terminal. The normal sequence for
	// starting a remote shell is as follows:
	// 1) Connect
	// 2) Authenticate
	// 3) OpenSessionChannel
	// 4) Request a PTY via this method if necessary.
	// 5) Start a shell by calling SendReqShell
	// 
	bool SendReqPty(int channelNum, const wchar_t *termType, int widthInChars, int heightInChars, int widthInPixels, int heightInPixels);

	// Creates an asynchronous task to call the SendReqPty method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqPtyAsync(int channelNum, const wchar_t *termType, int widthInChars, int heightInChars, int widthInPixels, int heightInPixels);

	// Sets an environment variable in the remote shell.
	bool SendReqSetEnv(int channelNum, const wchar_t *name, const wchar_t *value);

	// Creates an asynchronous task to call the SendReqSetEnv method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqSetEnvAsync(int channelNum, const wchar_t *name, const wchar_t *value);

	// Starts a shell on an open session channel. Some SSH servers require that a PTY
	// (pseudo-terminal) first be requested prior to starting a shell. In that case,
	// call SendReqPty prior to calling this method. Once a shell is started, commands
	// may be sent by calling ChannelSendString. (Don't forget to terminate commands
	// with a CRLF).
	bool SendReqShell(int channelNum);

	// Creates an asynchronous task to call the SendReqShell method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqShellAsync(int channelNum);

	// Delivers a signal to the remote process/service. signalName is one of the following:
	// ABRT, ALRM, FPE, HUP, ILL, INT, KILL, PIPE, QUIT, SEGV, TERM, USR1, USR2.
	// (Obviously, these are UNIX signals, so the remote SSH server would need to be a
	// Unix/Linux system.)
	bool SendReqSignal(int channelNum, const wchar_t *signalName);

	// Creates an asynchronous task to call the SendReqSignal method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqSignalAsync(int channelNum, const wchar_t *signalName);

	// Executes a pre-defined subsystem. The SFTP protocol (Secure File Transfer
	// Protocol) is started by the Chilkat SFTP component by starting the "sftp"
	// subsystem.
	bool SendReqSubsystem(int channelNum, const wchar_t *subsystemName);

	// Creates an asynchronous task to call the SendReqSubsystem method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqSubsystemAsync(int channelNum, const wchar_t *subsystemName);

	// When the client-side window (terminal) size changes, this message may be sent to
	// the server to inform it of the new size.
	bool SendReqWindowChange(int channelNum, int widthInChars, int heightInRows, int pixWidth, int pixHeight);

	// Creates an asynchronous task to call the SendReqWindowChange method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqWindowChangeAsync(int channelNum, int widthInChars, int heightInRows, int pixWidth, int pixHeight);

	// Allows the client to send an X11 forwarding request to the server. Chilkat only
	// provides this functionality because it is a message defined in the SSH
	// connection protocol. Chilkat has no advice for when or why it would be needed.
	bool SendReqX11Forwarding(int channelNum, bool singleConnection, const wchar_t *authProt, const wchar_t *authCookie, int screenNum);

	// Creates an asynchronous task to call the SendReqX11Forwarding method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqX11ForwardingAsync(int channelNum, bool singleConnection, const wchar_t *authProt, const wchar_t *authCookie, int screenNum);

	// This method should be ignored and not used.
	bool SendReqXonXoff(int channelNum, bool clientCanDo);

	// Creates an asynchronous task to call the SendReqXonXoff method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *SendReqXonXoffAsync(int channelNum, bool clientCanDo);

	// Sets a TTY mode that is included in the SendReqPty method call. Most commonly,
	// it is not necessary to call this method at all. Chilkat has no recommendations
	// or expertise as to why or when a particular mode might be useful. This
	// capability is provided because it is defined in the SSH connection protocol
	// specification.
	// 
	// This method can be called multiple times to set many terminal mode flags (one
	// per call).
	// 
	// The ttyValue is an integer, typically 0 or 1. Valid ttyName flag names include: VINTR,
	// VQUIT, VERASE, VKILL, VEOF, VEOL, VEOL2, VSTART, VSTOP, VSUSP, VDSUSP, VREPRINT,
	// VWERASE, VLNEXT, VFLUSH, VSWTCH, VSTATUS, VDISCARD, IGNPAR, PARMRK, INPCK,
	// ISTRIP, INLCR, IGNCR, ICRNL, IUCLC, IXON, IXANY, IXOFF, IMAXBEL, ISIG, ICANON,
	// XCASE, ECHO, ECHOE, ECHOK, ECHONL, NOFLSH, TOSTOP, IEXTEN, ECHOCTL, ECHOKE,
	// PENDIN, OPOST, OLCUC, ONLCR, OCRNL, ONOCR, ONLRET, CS7, CS8, PARENB, PARODD,
	// TTY_OP_ISPEED, TTY_OP_OSPEED
	// 
	bool SetTtyMode(const wchar_t *ttyName, int ttyValue);

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

	// Unlocks the component. This must be called once prior to calling any other
	// method. A fully-functional 30-day trial is automatically started when an
	// arbitrary string is passed to this method. For example, passing "Hello", or
	// "abc123" will unlock the component for the 1st thirty days after the initial
	// install.
	bool UnlockComponent(const wchar_t *unlockCode);

	// The pollTimeoutMs is the number of milliseconds to wait. To poll, pass a value of 0 in
	// pollTimeoutMs. Waits for an incoming message on any channel. This includes data, EOF,
	// CLOSE, etc. If a message arrives in the alotted time, the channel number is
	// returned. A value of -1 is returned for a timeout, and -2 for any other errors
	// such as if the connection is lost.
	// 
	// Note: If a channel number is returned, the message must still be read by calling
	// a method such as ChannelRead, ChannelReceiveUntilMatch, etc. Once the message is
	// actually received, it may be collected by calling GetReceivedText,
	// GetReceivedData, etc.
	// 
	int WaitForChannelMessage(int pollTimeoutMs);

	// Creates an asynchronous task to call the WaitForChannelMessage method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WaitForChannelMessageAsync(int pollTimeoutMs);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
