// CkStreamW.h: interface for the CkStreamW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkStreamW_H
#define _CkStreamW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkTaskW;
class CkBinDataW;
class CkByteData;
class CkStringBuilderW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkStreamW
class CK_VISIBLE_PUBLIC CkStreamW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkStreamW(const CkStreamW &);
	CkStreamW &operator=(const CkStreamW &);

    public:
	CkStreamW(void);
	virtual ~CkStreamW(void);

	

	static CkStreamW *createNew(void);
	

	CkStreamW(bool bCallbackOwned);
	static CkStreamW *createNew(bool bCallbackOwned);

	
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

	// true if the stream supports reading. Otherwise false.
	// 
	// Note: A stream that supports reading, which has already reached the
	// end-of-stream, will still have a CanRead value of true. This property
	// indicates the stream's inherent ability, and not whether or not the stream can
	// be read at a particular moment in time.
	// 
	bool get_CanRead(void);

	// true if the stream supports writing. Otherwise false.
	// 
	// Note: A stream that supports writing, which has already been closed for write,
	// will still have a CanWrite value of true. This property indicates the stream's
	// inherent ability, and not whether or not the stream can be written at a
	// particular moment in time.
	// 
	bool get_CanWrite(void);

	// true if it is known for sure that data is ready and waiting to be read.
	// false if it is not known for sure (it may be that data is immediately
	// available, but reading the stream with a ReadTimeoutMs of 0, which is to poll
	// the stream, is the way to find out).
	bool get_DataAvailable(void);

	// The default internal chunk size for reading or writing. The default value is
	// 65536. If this property is set to 0, it will cause the default chunk size
	// (65536) to be used. Note: The chunk size can have significant performance
	// impact. If performance is an issue, be sure to experiment with different chunk
	// sizes.
	int get_DefaultChunkSize(void);
	// The default internal chunk size for reading or writing. The default value is
	// 65536. If this property is set to 0, it will cause the default chunk size
	// (65536) to be used. Note: The chunk size can have significant performance
	// impact. If performance is an issue, be sure to experiment with different chunk
	// sizes.
	void put_DefaultChunkSize(int newVal);

	// true if the end-of-stream has already been reached. When the stream has
	// already ended, all calls to Read* methods will return false with the
	// ReadFailReason set to 3 (already at end-of-stream).
	bool get_EndOfStream(void);

	// true if the stream is closed for writing. Once closed, no more data may be
	// written to the stream.
	bool get_IsWriteClosed(void);

	// The length (in bytes) of the stream's source. If unknown, then this property
	// will have a value of -1. This property may be set by the application if it knows
	// in advance the length of the stream.
	__int64 get_Length(void);
	// The length (in bytes) of the stream's source. If unknown, then this property
	// will have a value of -1. This property may be set by the application if it knows
	// in advance the length of the stream.
	void put_Length(__int64 newVal);

	// The length (in bytes) of the stream's source. If unknown, then this property
	// will have a value of -1. This property may be set by the application if it knows
	// in advance the length of the stream.
	// 
	// Setting this property also sets the Length property (which is a 64-bit integer).
	// 
	int get_Length32(void);
	// The length (in bytes) of the stream's source. If unknown, then this property
	// will have a value of -1. This property may be set by the application if it knows
	// in advance the length of the stream.
	// 
	// Setting this property also sets the Length property (which is a 64-bit integer).
	// 
	void put_Length32(int newVal);

	// The number of bytes received by the stream.
	__int64 get_NumReceived(void);

	// The number of bytes sent by the stream.
	__int64 get_NumSent(void);

	// This property is automatically set when a Read* method is called. It indicates
	// the reason for failure. Possible values are:
	//     No failure (success)
	//     Timeout, or no data is immediately available for a polling read.
	//     Aborted by an application callback.
	//     Already at end-of-stream.
	//     Fatal stream error.
	//     Out-of-memory error (this is very unlikely).
	int get_ReadFailReason(void);

	// The maximum number of seconds to wait while reading. The default value is 30
	// seconds (i.e. 30000ms). A value of 0 indicates a poll. (A polling read is to
	// return with a timeout if no data is immediately available.)
	// 
	// Important: For most Chilkat timeout related properties, a value of 0 indicates
	// an infinite timeout. For this property, a value of 0 indicates a poll. If
	// setting a timeout related property (or method argument) to zero, be sure to
	// understand if 0 means "wait forever" or "poll".
	// 
	// The timeout value is not a total timeout. It is the maximum time to wait while
	// no additional data is forthcoming.
	// 
	int get_ReadTimeoutMs(void);
	// The maximum number of seconds to wait while reading. The default value is 30
	// seconds (i.e. 30000ms). A value of 0 indicates a poll. (A polling read is to
	// return with a timeout if no data is immediately available.)
	// 
	// Important: For most Chilkat timeout related properties, a value of 0 indicates
	// an infinite timeout. For this property, a value of 0 indicates a poll. If
	// setting a timeout related property (or method argument) to zero, be sure to
	// understand if 0 means "wait forever" or "poll".
	// 
	// The timeout value is not a total timeout. It is the maximum time to wait while
	// no additional data is forthcoming.
	// 
	void put_ReadTimeoutMs(int newVal);

	// Sets the sink to the path of a file. The file does not need to exist at the time
	// of setting this property. The sink file will be automatically opened on demand,
	// when the stream is first written.
	// 
	// Note: This property takes priority over other potential sinks. Make sure this
	// property is set to an empty string if the stream's sink is to be something else.
	// 
	void get_SinkFile(CkString &str);
	// Sets the sink to the path of a file. The file does not need to exist at the time
	// of setting this property. The sink file will be automatically opened on demand,
	// when the stream is first written.
	// 
	// Note: This property takes priority over other potential sinks. Make sure this
	// property is set to an empty string if the stream's sink is to be something else.
	// 
	const wchar_t *sinkFile(void);
	// Sets the sink to the path of a file. The file does not need to exist at the time
	// of setting this property. The sink file will be automatically opened on demand,
	// when the stream is first written.
	// 
	// Note: This property takes priority over other potential sinks. Make sure this
	// property is set to an empty string if the stream's sink is to be something else.
	// 
	void put_SinkFile(const wchar_t *newVal);

	// If true, the stream appends to the SinkFile rather than truncating and
	// re-writing the sink file. The default value is false.
	bool get_SinkFileAppend(void);
	// If true, the stream appends to the SinkFile rather than truncating and
	// re-writing the sink file. The default value is false.
	void put_SinkFileAppend(bool newVal);

	// Sets the source to the path of a file. The file does not need to exist at the
	// time of setting this property. The source file will be automatically opened on
	// demand, when the stream is first read.
	// 
	// Note: This property takes priority over other potential sources. Make sure this
	// property is set to an empty string if the stream's source is to be something
	// else.
	// 
	void get_SourceFile(CkString &str);
	// Sets the source to the path of a file. The file does not need to exist at the
	// time of setting this property. The source file will be automatically opened on
	// demand, when the stream is first read.
	// 
	// Note: This property takes priority over other potential sources. Make sure this
	// property is set to an empty string if the stream's source is to be something
	// else.
	// 
	const wchar_t *sourceFile(void);
	// Sets the source to the path of a file. The file does not need to exist at the
	// time of setting this property. The source file will be automatically opened on
	// demand, when the stream is first read.
	// 
	// Note: This property takes priority over other potential sources. Make sure this
	// property is set to an empty string if the stream's source is to be something
	// else.
	// 
	void put_SourceFile(const wchar_t *newVal);

	// If the source is a file, then this property can be used to stream one part of
	// the file. The SourceFilePartSize property defines the size (in bytes) of each
	// part. The SourceFilePart and SourceFilePartSize have default values of 0, which
	// means the entire SourceFile is streamed.
	// 
	// This property is a 0-based index. For example, if the SourceFilePartSize is
	// 1000, then part 0 is for bytes 0 to 999, part 1 is for bytes 1000 to 1999, etc.
	// 
	int get_SourceFilePart(void);
	// If the source is a file, then this property can be used to stream one part of
	// the file. The SourceFilePartSize property defines the size (in bytes) of each
	// part. The SourceFilePart and SourceFilePartSize have default values of 0, which
	// means the entire SourceFile is streamed.
	// 
	// This property is a 0-based index. For example, if the SourceFilePartSize is
	// 1000, then part 0 is for bytes 0 to 999, part 1 is for bytes 1000 to 1999, etc.
	// 
	void put_SourceFilePart(int newVal);

	// If the source is a file, then this property, in conjunction with the
	// SourceFilePart property, can be used to stream a single part of the file. This
	// property defines the size (in bytes) of each part. The SourceFilePart and
	// SourceFilePartSize have default values of 0, which means that by default, the
	// entire SourceFile is streamed.
	int get_SourceFilePartSize(void);
	// If the source is a file, then this property, in conjunction with the
	// SourceFilePart property, can be used to stream a single part of the file. This
	// property defines the size (in bytes) of each part. The SourceFilePart and
	// SourceFilePartSize have default values of 0, which means that by default, the
	// entire SourceFile is streamed.
	void put_SourceFilePartSize(int newVal);

	// If true, then include the BOM when creating a string source via
	// SetSourceString where the charset is utf-8, utf-16, etc. (The term "BOM" stands
	// for Byte Order Mark, also known as the preamble.) Also, if true, then include
	// the BOM when writing a string via the WriteString method. The default value of
	// this property is false.
	bool get_StringBom(void);
	// If true, then include the BOM when creating a string source via
	// SetSourceString where the charset is utf-8, utf-16, etc. (The term "BOM" stands
	// for Byte Order Mark, also known as the preamble.) Also, if true, then include
	// the BOM when writing a string via the WriteString method. The default value of
	// this property is false.
	void put_StringBom(bool newVal);

	// Indicates the expected character encoding, such as utf-8, windows-1256, utf-16,
	// etc. for methods that read text such as: ReadString, ReadToCRLF, ReadUntilMatch.
	// Also controls the character encoding when writing strings with the WriteString
	// method. The supported charsets are indicated at the link below.
	// 
	// The default value is "utf-8".
	// 
	void get_StringCharset(CkString &str);
	// Indicates the expected character encoding, such as utf-8, windows-1256, utf-16,
	// etc. for methods that read text such as: ReadString, ReadToCRLF, ReadUntilMatch.
	// Also controls the character encoding when writing strings with the WriteString
	// method. The supported charsets are indicated at the link below.
	// 
	// The default value is "utf-8".
	// 
	const wchar_t *stringCharset(void);
	// Indicates the expected character encoding, such as utf-8, windows-1256, utf-16,
	// etc. for methods that read text such as: ReadString, ReadToCRLF, ReadUntilMatch.
	// Also controls the character encoding when writing strings with the WriteString
	// method. The supported charsets are indicated at the link below.
	// 
	// The default value is "utf-8".
	// 
	void put_StringCharset(const wchar_t *newVal);

	// This property is automatically set when a Write* method is called. It indicates
	// the reason for failure. Possible values are:
	//     No failure (success)
	//     Timeout, or unable to immediately write when the WriteTimeoutMs is 0.
	//     Aborted by an application callback.
	//     The stream has already ended.
	//     Fatal stream error.
	//     Out-of-memory error (this is very unlikely).
	int get_WriteFailReason(void);

	// The maximum number of seconds to wait while writing. The default value is 30
	// seconds (i.e. 30000ms). A value of 0 indicates to return immediately if it is
	// not possible to write to the sink immediately.
	int get_WriteTimeoutMs(void);
	// The maximum number of seconds to wait while writing. The default value is 30
	// seconds (i.e. 30000ms). A value of 0 indicates to return immediately if it is
	// not possible to write to the sink immediately.
	void put_WriteTimeoutMs(int newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Loads the caller of the task's async method.
	bool LoadTaskCaller(CkTaskW &task);

	// Read as much data as is immediately available on the stream. If no data is
	// immediately available, it waits up to ReadTimeoutMs milliseconds for data to
	// arrive. The incoming data is appended to binData.
	bool ReadBd(CkBinDataW &binData);

	// Creates an asynchronous task to call the ReadBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadBdAsync(CkBinDataW &binData);

	// Read as much data as is immediately available on the stream. If no data is
	// immediately available, it waits up to ReadTimeoutMs milliseconds for data to
	// arrive.
	bool ReadBytes(CkByteData &outBytes);

	// Creates an asynchronous task to call the ReadBytes method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadBytesAsync(void);

	// The same as ReadBytes, except returns the received bytes in encoded string form.
	// The encoding argument indicates the encoding, which can be "base64", "hex", or any
	// of the multitude of encodings indicated in the link below.
	bool ReadBytesENC(const wchar_t *encoding, CkString &outStr);
	// The same as ReadBytes, except returns the received bytes in encoded string form.
	// The encoding argument indicates the encoding, which can be "base64", "hex", or any
	// of the multitude of encodings indicated in the link below.
	const wchar_t *readBytesENC(const wchar_t *encoding);

	// Creates an asynchronous task to call the ReadBytesENC method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadBytesENCAsync(const wchar_t *encoding);

	// Reads exactly numBytes bytes from the stream. If no data is immediately available,
	// it waits up to ReadTimeoutMs milliseconds for data to arrive.
	bool ReadNBytes(int numBytes, CkByteData &outBytes);

	// Creates an asynchronous task to call the ReadNBytes method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadNBytesAsync(int numBytes);

	// The same as ReadNBytes, except returns the received bytes in encoded string
	// form. The encoding argument indicates the encoding, which can be "base64", "hex", or
	// any of the multitude of encodings indicated in the link below.
	bool ReadNBytesENC(int numBytes, const wchar_t *encoding, CkString &outStr);
	// The same as ReadNBytes, except returns the received bytes in encoded string
	// form. The encoding argument indicates the encoding, which can be "base64", "hex", or
	// any of the multitude of encodings indicated in the link below.
	const wchar_t *readNBytesENC(int numBytes, const wchar_t *encoding);

	// Creates an asynchronous task to call the ReadNBytesENC method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadNBytesENCAsync(int numBytes, const wchar_t *encoding);

	// Read as much data as is immediately available on the stream. If no data is
	// immediately available, it waits up to ReadTimeoutMs milliseconds for data to
	// arrive. The data is appended to sb. The incoming bytes are interpreted
	// according to the StringCharset property. For example, if utf-8 bytes are
	// expected, then StringCharset should be set to "utf-8" prior to calling ReadSb.
	bool ReadSb(CkStringBuilderW &sb);

	// Creates an asynchronous task to call the ReadSb method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadSbAsync(CkStringBuilderW &sb);

	// Read as much data as is immediately available on the stream. If no data is
	// immediately available, it waits up to ReadTimeoutMs milliseconds for data to
	// arrive. The data is returned as a string. The incoming bytes are interpreted
	// according to the StringCharset property. For example, if utf-8 bytes are
	// expected, then StringCharset should be set to "utf-8" prior to calling
	// ReadString.
	bool ReadString(CkString &outStr);
	// Read as much data as is immediately available on the stream. If no data is
	// immediately available, it waits up to ReadTimeoutMs milliseconds for data to
	// arrive. The data is returned as a string. The incoming bytes are interpreted
	// according to the StringCharset property. For example, if utf-8 bytes are
	// expected, then StringCharset should be set to "utf-8" prior to calling
	// ReadString.
	const wchar_t *readString(void);

	// Creates an asynchronous task to call the ReadString method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadStringAsync(void);

	// Reads the stream until a CRLF is received. If no data is immediately available,
	// it waits up to ReadTimeoutMs milliseconds for data to arrive. The data is
	// returned as a string. The incoming bytes are interpreted according to the
	// StringCharset property. For example, if utf-8 bytes are expected, then
	// StringCharset should be set to "utf-8" prior to calling ReadString.
	// 
	// Note: If the end-of-stream is reached prior to receiving the CRLF, then the
	// remaining data is returned, and the ReadFailReason property will be set to 3 (to
	// indicate end-of-file). This is the only case where as string would be returned
	// that does not end with CRLF.
	// 
	bool ReadToCRLF(CkString &outStr);
	// Reads the stream until a CRLF is received. If no data is immediately available,
	// it waits up to ReadTimeoutMs milliseconds for data to arrive. The data is
	// returned as a string. The incoming bytes are interpreted according to the
	// StringCharset property. For example, if utf-8 bytes are expected, then
	// StringCharset should be set to "utf-8" prior to calling ReadString.
	// 
	// Note: If the end-of-stream is reached prior to receiving the CRLF, then the
	// remaining data is returned, and the ReadFailReason property will be set to 3 (to
	// indicate end-of-file). This is the only case where as string would be returned
	// that does not end with CRLF.
	// 
	const wchar_t *readToCRLF(void);

	// Creates an asynchronous task to call the ReadToCRLF method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadToCRLFAsync(void);

	// Reads the stream until the string indicated by matchStr is received. If no data is
	// immediately available, it waits up to ReadTimeoutMs milliseconds for data to
	// arrive. The data is returned as a string. The incoming bytes are interpreted
	// according to the StringCharset property. For example, if utf-8 bytes are
	// expected, then StringCharset should be set to "utf-8" prior to calling
	// ReadString.
	// 
	// Note: If the end-of-stream is reached prior to receiving the match string, then
	// the remaining data is returned, and the ReadFailReason property will be set to 3
	// (to indicate end-of-file). This is the only case where as string would be
	// returned that does not end with the desired match string.
	// 
	bool ReadUntilMatch(const wchar_t *matchStr, CkString &outStr);
	// Reads the stream until the string indicated by matchStr is received. If no data is
	// immediately available, it waits up to ReadTimeoutMs milliseconds for data to
	// arrive. The data is returned as a string. The incoming bytes are interpreted
	// according to the StringCharset property. For example, if utf-8 bytes are
	// expected, then StringCharset should be set to "utf-8" prior to calling
	// ReadString.
	// 
	// Note: If the end-of-stream is reached prior to receiving the match string, then
	// the remaining data is returned, and the ReadFailReason property will be set to 3
	// (to indicate end-of-file). This is the only case where as string would be
	// returned that does not end with the desired match string.
	// 
	const wchar_t *readUntilMatch(const wchar_t *matchStr);

	// Creates an asynchronous task to call the ReadUntilMatch method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *ReadUntilMatchAsync(const wchar_t *matchStr);

	// Resets the stream. If a source or sink is open, then it is closed. Properties
	// such as EndOfStream and IsWriteClose are reset to default values.
	void Reset(void);

	// Runs the stream to completion. This method should only be called when the source
	// of the string has been set by any of the following methods: SetSourceBytes,
	// SetSourceString, or SetSourceStream, or when the SourceFile property has been
	// set (giving the stream a file source).
	// 
	// This method will read the stream source and forward to the sink until the
	// end-of-stream is reached, and all data has been written to the sink.
	// 
	bool RunStream(void);

	// Creates an asynchronous task to call the RunStream method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *RunStreamAsync(void);

	// Sets the stream's sink to strm. Any data written to this stream's sink will
	// become available to strm on its source.
	bool SetSinkStream(CkStreamW &strm);

	// Sets the stream's source to the contents of sourceData.
	bool SetSourceBytes(CkByteData &sourceData);

	// Sets the stream's source to be the sink of strm. Any data written to strm's sink
	// will become available on this stream's source.
	bool SetSourceStream(CkStreamW &strm);

	// Sets the stream's source to the contents of srcStr. The charset indicates the
	// character encoding to be used for the byte representation of the srcStr.
	bool SetSourceString(const wchar_t *srcStr, const wchar_t *charset);

	// Writes the contents of binData to the stream.
	bool WriteBd(CkBinDataW &binData);

	// Creates an asynchronous task to call the WriteBd method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteBdAsync(CkBinDataW &binData);

	// Writes a single byte to the stream. The byteVal must have a value from 0 to 255.
	bool WriteByte(int byteVal);

	// Creates an asynchronous task to call the WriteByte method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteByteAsync(int byteVal);

	// Writes binary bytes to a stream.
	bool WriteBytes(CkByteData &byteData);

	// Creates an asynchronous task to call the WriteBytes method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteBytesAsync(CkByteData &byteData);

	// Writes binary bytes to a stream.
	bool WriteBytes2(const void *pByteData, unsigned long szByteData);

	// Writes binary bytes to a stream. The byte data is passed in encoded string form,
	// where the encoding can be "base64", "hex", or any of the supported binary encodings
	// listed at the link below.
	bool WriteBytesENC(const wchar_t *byteData, const wchar_t *encoding);

	// Creates an asynchronous task to call the WriteBytesENC method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteBytesENCAsync(const wchar_t *byteData, const wchar_t *encoding);

	// Indicates that no more data will be written to the stream.
	bool WriteClose(void);

	// Writes the contents of sb to the stream. The actual bytes written are the byte
	// representation of the string as indicated by the StringCharset property. For
	// example, to write utf-8 bytes, first set StringCharset equal to "utf-8" and then
	// call WriteSb.
	bool WriteSb(CkStringBuilderW &sb);

	// Creates an asynchronous task to call the WriteSb method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteSbAsync(CkStringBuilderW &sb);

	// Writes a string to a stream. The actual bytes written are the byte
	// representation of the string as indicated by the StringCharset property. For
	// example, to write utf-8 bytes, first set StringCharset equal to "utf-8" and then
	// call WriteString.
	bool WriteString(const wchar_t *str);

	// Creates an asynchronous task to call the WriteString method with the arguments
	// provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *WriteStringAsync(const wchar_t *str);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
