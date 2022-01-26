// CkWebSocket.h: interface for the CkWebSocket class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkWebSocket_H
#define _CkWebSocket_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacks.h"

class CkBinData;
class CkStringBuilder;
class CkTask;
class CkRest;
class CkBaseProgress;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkWebSocket
class CK_VISIBLE_PUBLIC CkWebSocket  : public CkClassWithCallbacks
{
    private:

	// Don't allow assignment or copying these objects.
	CkWebSocket(const CkWebSocket &);
	CkWebSocket &operator=(const CkWebSocket &);

    public:
	CkWebSocket(void);
	virtual ~CkWebSocket(void);

	static CkWebSocket *createNew(void);
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
	// If true, then a Close control frame is automatically sent in response to
	// receiving a Close control frame (assuming that we did not initiate the Close in
	// the first place). When the Close frame has both been received and sent, the
	// underlying connection is automatically closed (as per the WebSocket protocol RFC
	// specifications). Thus, if this property is true, then two things automatically
	// happen when a Close frame is received: (1) a Close frame is sent in response,
	// and (2) the connection is closed.
	// 
	// The default value of this property is true.
	// 
	bool get_CloseAutoRespond(void);
	// If true, then a Close control frame is automatically sent in response to
	// receiving a Close control frame (assuming that we did not initiate the Close in
	// the first place). When the Close frame has both been received and sent, the
	// underlying connection is automatically closed (as per the WebSocket protocol RFC
	// specifications). Thus, if this property is true, then two things automatically
	// happen when a Close frame is received: (1) a Close frame is sent in response,
	// and (2) the connection is closed.
	// 
	// The default value of this property is true.
	// 
	void put_CloseAutoRespond(bool newVal);

	// The reason string received with the Close frame, if any.
	void get_CloseReason(CkString &str);
	// The reason string received with the Close frame, if any.
	const char *closeReason(void);

	// If true, then a Close frame was already received on this websocket connection.
	// If CloseAutoRespond is false, then an application can check this property
	// value to determine if a Close frame should be sent in response.
	bool get_CloseReceived(void);

	// The status code received with the Close frame. If no status code was provided,
	// or no Close frame has yet been received, then this property will be 0.
	int get_CloseStatusCode(void);

	// true if the last data frame received by calling ReadFrame was a final frame.
	// Otherwise false.
	bool get_FinalFrame(void);

	// The number of bytes accumulated from one or more calls to ReadFrame. Accumulated
	// incoming frame data can be retrieved by calling GetFrameData, GetFrameDataSb, or
	// GetFrameDataBd.
	int get_FrameDataLen(void);

	// Indicates the type of frame received in the last call to ReadFrame. Possible
	// values are "Continuation", "Text", "Binary", "Close", "Ping", or "Pong".
	// Initially this property is set to the empty string because nothing has yet been
	// received.
	void get_FrameOpcode(CkString &str);
	// Indicates the type of frame received in the last call to ReadFrame. Possible
	// values are "Continuation", "Text", "Binary", "Close", "Ping", or "Pong".
	// Initially this property is set to the empty string because nothing has yet been
	// received.
	const char *frameOpcode(void);

	// The integer value of the opcode (type of frame) received in the last call to
	// ReadFrame. Possible values are:
	// 0 - Continuation
	// 1 - Text
	// 2 - Binary
	// 8 - Close
	// 9 - Ping
	// 10 - Pong
	int get_FrameOpcodeInt(void);

	// The maximum amount of time to wait for additional incoming data when receiving,
	// or the max time to wait to send additional data. The default value is 30000 (30
	// seconds). This is not an overall max timeout. Rather, it is the maximum time to
	// wait when receiving or sending has halted.
	int get_IdleTimeoutMs(void);
	// The maximum amount of time to wait for additional incoming data when receiving,
	// or the max time to wait to send additional data. The default value is 30000 (30
	// seconds). This is not an overall max timeout. Rather, it is the maximum time to
	// wait when receiving or sending has halted.
	void put_IdleTimeoutMs(int newVal);

	// Returns true if the websocket is connected. Otherwise returns false.
	bool get_IsConnected(void);

	// If true, then a Ping frame was received, but no Pong frame has yet been sent
	// in response. The application should send a Pong frame by calling SendPong as
	// soon as possible.
	bool get_NeedSendPong(void);

	// If true, then a Pong frame is automatically sent when a Ping frame is
	// received. If set to false, then the application may check the NeedSendPong
	// property to determine if a Pong response is needed, and if so, may call the
	// SendPong method to send a Pong.
	// 
	// Note: If this property is true, then the ReadFrame method will auto-consume
	// incoming Ping frames. In other words, ReadFrame will continue with reading the
	// next incoming frame (thus Ping frames will never be returned to the
	// application). This relieves the application from having to worry about receiving
	// and handling spurious Ping frames.
	// 
	// The default value is true.
	// 
	bool get_PingAutoRespond(void);
	// If true, then a Pong frame is automatically sent when a Ping frame is
	// received. If set to false, then the application may check the NeedSendPong
	// property to determine if a Pong response is needed, and if so, may call the
	// SendPong method to send a Pong.
	// 
	// Note: If this property is true, then the ReadFrame method will auto-consume
	// incoming Ping frames. In other words, ReadFrame will continue with reading the
	// next incoming frame (thus Ping frames will never be returned to the
	// application). This relieves the application from having to worry about receiving
	// and handling spurious Ping frames.
	// 
	// The default value is true.
	// 
	void put_PingAutoRespond(bool newVal);

	// If true, then incoming Pong frames are automatically consumed, and a call to
	// ReadFrame will continue reading until it receives a non-Pong frame. The
	// PongConsumed property can be checked to see if the last ReadFrame method call
	// auto-consumed a Pong frame.
	// 
	// The default value is true.
	// 
	bool get_PongAutoConsume(void);
	// If true, then incoming Pong frames are automatically consumed, and a call to
	// ReadFrame will continue reading until it receives a non-Pong frame. The
	// PongConsumed property can be checked to see if the last ReadFrame method call
	// auto-consumed a Pong frame.
	// 
	// The default value is true.
	// 
	void put_PongAutoConsume(bool newVal);

	// Is true if the last call to ReadFrame auto-consumed a Pong frame. This
	// property is reset to false each time a ReadFrame method is called, and will
	// get set to true if (1) the PongAutoConsume property is true and (2) a Pong
	// frame was consumed within the ReadFrame method.
	// 
	// The purpose of PongAutoConsume and PongConsumed is to eliminate the concern for
	// unanticipated Pong frames in the stream. In the websocket protocol, both sides
	// (client and server) may send Pong frames at any time. In addition, if a Ping
	// frame is sent, the corresponding Pong response frame can arrive at some
	// unanticipated point later in the conversation. It's also possible, if several
	// Ping frames are sent, that a Pong response frame is only sent for the most
	// recent Ping frame. The default behavior of Chilkat's WebSocket API is to
	// auto-consume incoming Pong frames and set this property to true. This allows
	// the application to call a ReadFrame method for whatever application data frame
	// it may be expecting, without needing to be concerned if the next incoming frame
	// is a Pong frame.
	// 
	bool get_PongConsumed(void);

	// If the ReadFrame method returns false, this property holds the fail reason. It
	// can have one of the following values:
	// 0 - No failure.
	// 1 - Read Timeout.
	// 2 - Aborted by Application Callback.
	// 3 - Fatal Socket Error (Lost Connection).
	// 4 - Received invalid WebSocket frame bytes.
	// 99 - A catch-all for any unknown failure.  (Should not ever occur.  If it does, contact Chilkat.)
	int get_ReadFrameFailReason(void);

	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty. Can be set to a
	// list of the following comma separated keywords:
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	void get_UncommonOptions(CkString &str);
	// This is a catch-all property to be used for uncommon needs. This property
	// defaults to the empty string and should typically remain empty. Can be set to a
	// list of the following comma separated keywords:
	//     "ProtectFromVpn" - Introduced in v9.5.0.80. On Android systems, will bypass
	//     any VPN that may be installed or active.
	const char *uncommonOptions(void);



	// ----------------------
	// Methods
	// ----------------------
	// Adds the required WebSocket client-side open handshake headers. The headers
	// specifically added to the previously specified REST object (in the call to
	// UseConnection) are:
	// Upgrade: websocket
	// Connection: Upgrade
	// Sec-WebSocket-Key: ...
	// Sec-WebSocket-Version: 13
	bool AddClientHeaders(void);


	// Forcibly closes the underlying connection. This is a non-clean way to close the
	// connection, but may be used if needed. The clean way to close a websocket is to
	// send a Close frame, and then receive the Close response.
	bool CloseConnection(void);


	// Returns the accumulated received frame data as a string. Calling GetFrameData
	// clears the internal receive buffer.
	bool GetFrameData(CkString &outStr);

	// Returns the accumulated received frame data as a string. Calling GetFrameData
	// clears the internal receive buffer.
	const char *getFrameData(void);
	// Returns the accumulated received frame data as a string. Calling GetFrameData
	// clears the internal receive buffer.
	const char *frameData(void);


	// Returns the accumulated received frame data in a BinData object. The received
	// data is appended to the binData.
	// 
	// Calling this method clears the internal receive buffer.
	// 
	bool GetFrameDataBd(CkBinData &binData);


	// Returns the accumulated received frame data in a StringBuilder object. The
	// received data is appended to the sb.
	// 
	// Calling this method clears the internal receive buffer.
	// 
	bool GetFrameDataSb(CkStringBuilder &sb);


	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTask &task);


	// Check to see if data is available for reading on the websocket. Returns true
	// if data is waiting and false if no data is waiting to be read.
	bool PollDataAvailable(void);


	// Reads a single frame from the connected websocket. If a frame was successfuly
	// received, then the following properties are set: FrameOpcode, FrameDataLen,
	// FinalFrame, and the received frame data can be retrieved by calling
	// GetFrameData, GetFrameDataSb, or GetFrameDataBd.
	bool ReadFrame(void);

	// Reads a single frame from the connected websocket. If a frame was successfuly
	// received, then the following properties are set: FrameOpcode, FrameDataLen,
	// FinalFrame, and the received frame data can be retrieved by calling
	// GetFrameData, GetFrameDataSb, or GetFrameDataBd.
	CkTask *ReadFrameAsync(void);


	// Sends a Close control frame. If includeStatus is true, then the statusCode is sent in the
	// application data part of the Close frame. A Close reason may be provided only if
	// includeStatus is true. If this Close was sent to satisfy an already-received Close
	// frame, then the underlying connection will also automatically be closed.
	// 
	// Note: If a status code and reason are provided, the utf-8 representation of the
	// reason string must be 123 bytes or less. Chilkat will automatically truncate the
	// reason to 123 bytes if necessary. Also, the status code must be an integer in the
	// range 0 to 16383.
	// 
	// The WebSocket protocol specifies some pre-defined status codes atWebSocket
	// Status Codes
	// <https://tools.ietf.org/html/rfc6455#section-7.4.1>. For a normal closure, a
	// status code value of 1000 should be used. The reason can be any string, as long
	// as it is 123 bytes or less.
	// 
	bool SendClose(bool includeStatus, int statusCode, const char *reason);

	// Sends a Close control frame. If includeStatus is true, then the statusCode is sent in the
	// application data part of the Close frame. A Close reason may be provided only if
	// includeStatus is true. If this Close was sent to satisfy an already-received Close
	// frame, then the underlying connection will also automatically be closed.
	// 
	// Note: If a status code and reason are provided, the utf-8 representation of the
	// reason string must be 123 bytes or less. Chilkat will automatically truncate the
	// reason to 123 bytes if necessary. Also, the status code must be an integer in the
	// range 0 to 16383.
	// 
	// The WebSocket protocol specifies some pre-defined status codes atWebSocket
	// Status Codes
	// <https://tools.ietf.org/html/rfc6455#section-7.4.1>. For a normal closure, a
	// status code value of 1000 should be used. The reason can be any string, as long
	// as it is 123 bytes or less.
	// 
	CkTask *SendCloseAsync(bool includeStatus, int statusCode, const char *reason);


	// Sends a single data frame containing a string. If this is the final frame in a
	// message, then finalFrame should be set to true. Otherwise set finalFrame equal to false.
	bool SendFrame(const char *stringToSend, bool finalFrame);

	// Sends a single data frame containing a string. If this is the final frame in a
	// message, then finalFrame should be set to true. Otherwise set finalFrame equal to false.
	CkTask *SendFrameAsync(const char *stringToSend, bool finalFrame);


	// Sends a single data frame containing binary data (the contents of bdToSend). If this
	// is the final frame in a message, then finalFrame should be set to true. Otherwise
	// set finalFrame equal to false.
	bool SendFrameBd(CkBinData &bdToSend, bool finalFrame);

	// Sends a single data frame containing binary data (the contents of bdToSend). If this
	// is the final frame in a message, then finalFrame should be set to true. Otherwise
	// set finalFrame equal to false.
	CkTask *SendFrameBdAsync(CkBinData &bdToSend, bool finalFrame);


	// Sends a single data frame containing a string (the contents of sbToSend). If this is
	// the final frame in a message, then finalFrame should be set to true. Otherwise set
	// finalFrame equal to false.
	bool SendFrameSb(CkStringBuilder &sbToSend, bool finalFrame);

	// Sends a single data frame containing a string (the contents of sbToSend). If this is
	// the final frame in a message, then finalFrame should be set to true. Otherwise set
	// finalFrame equal to false.
	CkTask *SendFrameSbAsync(CkStringBuilder &sbToSend, bool finalFrame);


	// Sends a Ping control frame, optionally including text data. If pingData is
	// non-empty, the utf-8 representation of the string must be 125 bytes or less.
	// Chilkat will automatically truncate the pingData to 125 bytes if necessary.
	bool SendPing(const char *pingData);

	// Sends a Ping control frame, optionally including text data. If pingData is
	// non-empty, the utf-8 representation of the string must be 125 bytes or less.
	// Chilkat will automatically truncate the pingData to 125 bytes if necessary.
	CkTask *SendPingAsync(const char *pingData);


	// Sends a Pong control frame. If this Pong frame is sent to satisfy an
	// unresponded-to Ping frame, then the previously received Ping data is
	// automatically sent in this Pong frame.
	bool SendPong(void);

	// Sends a Pong control frame. If this Pong frame is sent to satisfy an
	// unresponded-to Ping frame, then the previously received Ping data is
	// automatically sent in this Pong frame.
	CkTask *SendPongAsync(void);


	// Initializes the connection for a WebSocket session. All WebSocket sessions begin
	// with a call to UseConnection. A Chilkat REST object is used for the connection
	// because the WebSocket handshake begins with an HTTP GET request. The Chilkat
	// REST API provides the ability to add custom headers, authentication, etc. to the
	// opening GET handshake. It also provides the ability to establish connections
	// over TLS or SSH and to benefit from the rich set of features already present
	// relating to HTTP proxies, SOCKS proxies, bandwidth throttling, IPv6, socket
	// options, etc.
	bool UseConnection(CkRest &connection);


	// Called after sending the opening handshake from the Rest object. Validates the
	// server's response to the opening handshake message. If validation is successful,
	// the application may begin sending and receiving data and control frames.
	bool ValidateServerHandshake(void);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
