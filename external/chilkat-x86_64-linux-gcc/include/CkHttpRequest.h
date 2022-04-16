// CkHttpRequest.h: interface for the CkHttpRequest class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkHttpRequest_H
#define _CkHttpRequest_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkBinData;
class CkByteData;
class CkStringBuilder;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkHttpRequest
class CK_VISIBLE_PUBLIC CkHttpRequest  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkHttpRequest(const CkHttpRequest &);
	CkHttpRequest &operator=(const CkHttpRequest &);

    public:
	CkHttpRequest(void);
	virtual ~CkHttpRequest(void);

	static CkHttpRequest *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// Sets an explicit boundary string to be used in multipart/form-data requests. If
	// no Boundary is set, then a boundary string is automaticaly generated as needed
	// during the sending of a request.
	void get_Boundary(CkString &str);
	// Sets an explicit boundary string to be used in multipart/form-data requests. If
	// no Boundary is set, then a boundary string is automaticaly generated as needed
	// during the sending of a request.
	const char *boundary(void);
	// Sets an explicit boundary string to be used in multipart/form-data requests. If
	// no Boundary is set, then a boundary string is automaticaly generated as needed
	// during the sending of a request.
	void put_Boundary(const char *newVal);

	// Controls the character encoding used for HTTP request parameters for POST
	// requests. The default value is the ANSI charset of the computer. The charset
	// should match the charset expected by the form target.
	// 
	// The "charset" attribute is only included in the Content-Type header of the
	// request if the SendCharset property is set to true.
	// 
	void get_Charset(CkString &str);
	// Controls the character encoding used for HTTP request parameters for POST
	// requests. The default value is the ANSI charset of the computer. The charset
	// should match the charset expected by the form target.
	// 
	// The "charset" attribute is only included in the Content-Type header of the
	// request if the SendCharset property is set to true.
	// 
	const char *charset(void);
	// Controls the character encoding used for HTTP request parameters for POST
	// requests. The default value is the ANSI charset of the computer. The charset
	// should match the charset expected by the form target.
	// 
	// The "charset" attribute is only included in the Content-Type header of the
	// request if the SendCharset property is set to true.
	// 
	void put_Charset(const char *newVal);

	// The ContentType property sets the "Content-Type" header field, and identifies
	// the content-type of the HTTP request body. Common values are:
	// 
	//         
	// application/x-www-form-urlencoded    
	// multipart/form-data    
	// application/json    
	// application/xml    
	//     
	// 
	// If ContentType is set equal to the empty string, then no Content-Type header is
	// included in the HTTP request.
	void get_ContentType(CkString &str);
	// The ContentType property sets the "Content-Type" header field, and identifies
	// the content-type of the HTTP request body. Common values are:
	// 
	//         
	// application/x-www-form-urlencoded    
	// multipart/form-data    
	// application/json    
	// application/xml    
	//     
	// 
	// If ContentType is set equal to the empty string, then no Content-Type header is
	// included in the HTTP request.
	const char *contentType(void);
	// The ContentType property sets the "Content-Type" header field, and identifies
	// the content-type of the HTTP request body. Common values are:
	// 
	//         
	// application/x-www-form-urlencoded    
	// multipart/form-data    
	// application/json    
	// application/xml    
	//     
	// 
	// If ContentType is set equal to the empty string, then no Content-Type header is
	// included in the HTTP request.
	void put_ContentType(const char *newVal);

	// Composes and returns the entire MIME header of the HTTP request.
	void get_EntireHeader(CkString &str);
	// Composes and returns the entire MIME header of the HTTP request.
	const char *entireHeader(void);
	// Composes and returns the entire MIME header of the HTTP request.
	void put_EntireHeader(const char *newVal);

	// The HttpVerb property should be set to the name of the HTTP method that appears
	// on the "start line" of an HTTP request, such as GET, POST, PUT, DELETE, etc. It
	// is also possible to use the various WebDav verbs such as PROPFIND, PROPPATCH,
	// MKCOL, COPY, MOVE, LOCK, UNLOCK, etc. In general, the HttpVerb may be set to
	// anything, even custom verbs recognized by a custom server-side app.
	void get_HttpVerb(CkString &str);
	// The HttpVerb property should be set to the name of the HTTP method that appears
	// on the "start line" of an HTTP request, such as GET, POST, PUT, DELETE, etc. It
	// is also possible to use the various WebDav verbs such as PROPFIND, PROPPATCH,
	// MKCOL, COPY, MOVE, LOCK, UNLOCK, etc. In general, the HttpVerb may be set to
	// anything, even custom verbs recognized by a custom server-side app.
	const char *httpVerb(void);
	// The HttpVerb property should be set to the name of the HTTP method that appears
	// on the "start line" of an HTTP request, such as GET, POST, PUT, DELETE, etc. It
	// is also possible to use the various WebDav verbs such as PROPFIND, PROPPATCH,
	// MKCOL, COPY, MOVE, LOCK, UNLOCK, etc. In general, the HttpVerb may be set to
	// anything, even custom verbs recognized by a custom server-side app.
	void put_HttpVerb(const char *newVal);

	// The HTTP version in the request header. Defaults to "1.1".
	void get_HttpVersion(CkString &str);
	// The HTTP version in the request header. Defaults to "1.1".
	const char *httpVersion(void);
	// The HTTP version in the request header. Defaults to "1.1".
	void put_HttpVersion(const char *newVal);

	// Returns the number of request header fields.
	int get_NumHeaderFields(void);

	// Returns the number of query parameters.
	int get_NumParams(void);

	// The path of the resource requested. A path of "/" indicates the default document
	// of a domain.
	// 
	// Explaining the Parts of a URL
	// 
	// http://example.com:8042/over/there?name=ferret#nose
	// \__/   \______________/\_________/ \________/ \__/
	//  |           |            |            |        |
	// scheme   domain+port     path        query   fragment
	// 
	// This property should be set to the path part of the URL. You may also include
	// the query part in this property value. If the Content-Type of the request is NOT
	// application/x-www-form-urlencoded, then you would definitely want to include
	// query parameters in the path. If the Content-Type of the request IS
	// application/x-www-form-urlencoded, the query parameters are passed in the body
	// of the request. It is also possible to pass some query parameters via the path,
	// and some in the body of a application/x-www-form-urlencoded request, but you
	// shouldn't include the same parameter in both places. You would never need to
	// include the fragment part. The fragment is nothing more than an instruction for
	// a browser to automatically navigate to a particular location in the HTML page
	// (assuming the request returns HTML, otherwise a fragment makes no sense).
	// 
	void get_Path(CkString &str);
	// The path of the resource requested. A path of "/" indicates the default document
	// of a domain.
	// 
	// Explaining the Parts of a URL
	// 
	// http://example.com:8042/over/there?name=ferret#nose
	// \__/   \______________/\_________/ \________/ \__/
	//  |           |            |            |        |
	// scheme   domain+port     path        query   fragment
	// 
	// This property should be set to the path part of the URL. You may also include
	// the query part in this property value. If the Content-Type of the request is NOT
	// application/x-www-form-urlencoded, then you would definitely want to include
	// query parameters in the path. If the Content-Type of the request IS
	// application/x-www-form-urlencoded, the query parameters are passed in the body
	// of the request. It is also possible to pass some query parameters via the path,
	// and some in the body of a application/x-www-form-urlencoded request, but you
	// shouldn't include the same parameter in both places. You would never need to
	// include the fragment part. The fragment is nothing more than an instruction for
	// a browser to automatically navigate to a particular location in the HTML page
	// (assuming the request returns HTML, otherwise a fragment makes no sense).
	// 
	const char *path(void);
	// The path of the resource requested. A path of "/" indicates the default document
	// of a domain.
	// 
	// Explaining the Parts of a URL
	// 
	// http://example.com:8042/over/there?name=ferret#nose
	// \__/   \______________/\_________/ \________/ \__/
	//  |           |            |            |        |
	// scheme   domain+port     path        query   fragment
	// 
	// This property should be set to the path part of the URL. You may also include
	// the query part in this property value. If the Content-Type of the request is NOT
	// application/x-www-form-urlencoded, then you would definitely want to include
	// query parameters in the path. If the Content-Type of the request IS
	// application/x-www-form-urlencoded, the query parameters are passed in the body
	// of the request. It is also possible to pass some query parameters via the path,
	// and some in the body of a application/x-www-form-urlencoded request, but you
	// shouldn't include the same parameter in both places. You would never need to
	// include the fragment part. The fragment is nothing more than an instruction for
	// a browser to automatically navigate to a particular location in the HTML page
	// (assuming the request returns HTML, otherwise a fragment makes no sense).
	// 
	void put_Path(const char *newVal);

	// Controls whether the charset is explicitly included in the content-type header
	// field of the HTTP POST request. The default value of this property is false.
	bool get_SendCharset(void);
	// Controls whether the charset is explicitly included in the content-type header
	// field of the HTTP POST request. The default value of this property is false.
	void put_SendCharset(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Adds a file to an upload request where the contents of the file come from byteData.
	// 
	// name is an arbitrary name. (In HTML, it is the form field name of the input
	// tag.)
	// remoteFilename is the name of the file to be created on the HTTP server.
	// byteData contains the bytes to be uploaded.
	// contentType contains is the value of the Content-Type header. An empty string may be
	// passed to allow Chilkat to automatically determine the Content-Type based on the
	// filename extension.
	// 
	bool AddBdForUpload(const char *name, const char *remoteFilename, CkBinData &byteData, const char *contentType);


	// Adds a file to an upload request where the contents of the file come from an
	// in-memory byte array. To create a file upload request, set the ContentType
	// property = "multipart/form-data" and then call AddBytesForUpload,
	// AddStringForUpload, or AddFileForUpload for each file to be uploaded.
	// 
	// name is an arbitrary name. (In HTML, it is the form field name of the input
	// tag.)
	// remoteFileName is the name of the file to be created on the HTTP server.
	// byteData contains the contents (bytes) to be uploaded.
	// 
	bool AddBytesForUpload(const char *name, const char *remoteFileName, CkByteData &byteData);


	// Same as AddBytesForUpload, but allows the Content-Type header field to be
	// directly specified. (Otherwise, the Content-Type header is automatically
	// determined based on the remoteFileName's file extension.)
	bool AddBytesForUpload2(const char *name, const char *remoteFileName, CkByteData &byteData, const char *contentType);


	// Adds a file to an upload request. To create a file upload request, set the
	// ContentType property = "multipart/form-data" and then call AddFileForUpload,
	// AddBytesForUpload, or AddStringForUpload for each file to be uploaded. This
	// method does not read the file into memory. When the upload occurs, the data is
	// streamed directly from the file, thus allowing for very large files to be
	// uploaded without consuming large amounts of memory.
	// 
	// name is an arbitrary name. (In HTML, it is the form field name of the input
	// tag.)
	// filePath is the path to an existing file in the local filesystem.
	// 
	bool AddFileForUpload(const char *name, const char *filePath);


	// Same as AddFileForUpload, but allows the Content-Type header field to be
	// directly specified. (Otherwise, the Content-Type header is automatically
	// determined based on the file extension.)
	// 
	// name is an arbitrary name. (In HTML, it is the form field name of the input
	// tag.)
	// filePath is the path to an existing file in the local filesystem.
	// 
	bool AddFileForUpload2(const char *name, const char *filePath, const char *contentType);


	// Adds a request header to the HTTP request. If a header having the same field
	// name is already present, this method replaces it.
	// 
	// Note: Never explicitly set the Content-Length header field. Chilkat will
	// automatically compute the correct length and add the Content-Length header to
	// all POST, PUT, or any other request where the Content-Length needs to be
	// specified. (GET requests always have a 0 length body, and therefore never need a
	// Content-Length header field.)
	// 
	void AddHeader(const char *name, const char *value);


	// Computes the Amazon MWS signature using the mwsSecretKey and adds the "Signature"
	// parameter to the request. This method should be called for all Amazon
	// Marketplace Web Service (Amazon MWS) HTTP requests. It should be called after
	// all request parameters have been added.
	// 
	// Important: The Chilkat v9.5.0.75 release accidentally breaks Amazon MWS (not
	// AWS) authentication. If you need MWS with 9.5.0.75, send email to
	// support@chilkatsoft.com for a hotfix, or revert back to v9.5.0.73, or update to
	// a version after 9.5.0.75.
	// 
	// The domain should be the domain of the request, such as one of the following:
	//     mws.amazonservices.com
	//     mws-eu.amazonservices.com
	//     mws.amazonservices.in
	//     mws.amazonservices.com.cn
	//     mws.amazonservices.jp
	// 
	// Note: This method automatically adds or replaces the existing Timestamp
	// parameter to the current system date/time.
	// 
	bool AddMwsSignature(const char *domain, const char *mwsSecretKey);


	// Adds a request query parameter (name/value pair) to the HTTP request. The name
	// and value strings passed to this method should not be URL encoded.
	void AddParam(const char *name, const char *value);


	// Same as AddFileForUpload, but the upload data comes from an in-memory string
	// instead of a file.
	bool AddStringForUpload(const char *name, const char *filename, const char *strData, const char *charset);


	// Same as AddStringForUpload, but allows the Content-Type header field to be
	// directly specified. (Otherwise, the Content-Type header is automatically
	// determined based on the filename's file extension.)
	bool AddStringForUpload2(const char *name, const char *filename, const char *strData, const char *charset, const char *contentType);


	// Adds a request header to the Nth sub-header of the HTTP request. If a header
	// having the same field name is already present, this method replaces it.
	bool AddSubHeader(int index, const char *name, const char *value);


	// The same as GenerateRequestText, except the generated request is written to the
	// file specified by path.
	bool GenerateRequestFile(const char *path);


	// Returns the request text that would be sent if Http.SynchronousRequest was
	// called.
	bool GenerateRequestText(CkString &outStr);

	// Returns the request text that would be sent if Http.SynchronousRequest was
	// called.
	const char *generateRequestText(void);

	// Returns the value of a request header field.
	bool GetHeaderField(const char *name, CkString &outStr);

	// Returns the value of a request header field.
	const char *getHeaderField(const char *name);
	// Returns the value of a request header field.
	const char *headerField(const char *name);


	// Returns the Nth request header field name. Indexing begins at 0, and the number
	// of request header fields is specified by the NumHeaderFields property.
	bool GetHeaderName(int index, CkString &outStr);

	// Returns the Nth request header field name. Indexing begins at 0, and the number
	// of request header fields is specified by the NumHeaderFields property.
	const char *getHeaderName(int index);
	// Returns the Nth request header field name. Indexing begins at 0, and the number
	// of request header fields is specified by the NumHeaderFields property.
	const char *headerName(int index);


	// Returns the Nth request header field value. Indexing begins at 0, and the number
	// of request header fields is specified by the NumHeaderFields property.
	bool GetHeaderValue(int index, CkString &outStr);

	// Returns the Nth request header field value. Indexing begins at 0, and the number
	// of request header fields is specified by the NumHeaderFields property.
	const char *getHeaderValue(int index);
	// Returns the Nth request header field value. Indexing begins at 0, and the number
	// of request header fields is specified by the NumHeaderFields property.
	const char *headerValue(int index);


	// Returns a request query parameter value by name.
	bool GetParam(const char *name, CkString &outStr);

	// Returns a request query parameter value by name.
	const char *getParam(const char *name);
	// Returns a request query parameter value by name.
	const char *param(const char *name);


	// Returns the Nth request query parameter field name. Indexing begins at 0, and
	// the number of request query parameter fields is specified by the NumParams
	// property.
	bool GetParamName(int index, CkString &outStr);

	// Returns the Nth request query parameter field name. Indexing begins at 0, and
	// the number of request query parameter fields is specified by the NumParams
	// property.
	const char *getParamName(int index);
	// Returns the Nth request query parameter field name. Indexing begins at 0, and
	// the number of request query parameter fields is specified by the NumParams
	// property.
	const char *paramName(int index);


	// Returns the Nth request query parameter field value. Indexing begins at 0, and
	// the number of request query parameter fields is specified by the NumParams
	// property.
	bool GetParamValue(int index, CkString &outStr);

	// Returns the Nth request query parameter field value. Indexing begins at 0, and
	// the number of request query parameter fields is specified by the NumParams
	// property.
	const char *getParamValue(int index);
	// Returns the Nth request query parameter field value. Indexing begins at 0, and
	// the number of request query parameter fields is specified by the NumParams
	// property.
	const char *paramValue(int index);


	// Returns the request parameters in URL encoded form (i.e. in the exact form that
	// would be sent if the ContentType property was
	// application/x-www-form-urlencoded). For example, if a request has two params:
	// param1="abc 123" and param2="abc-123", then GetUrlEncodedParams would return
	// "abc+123
	bool GetUrlEncodedParams(CkString &outStr);

	// Returns the request parameters in URL encoded form (i.e. in the exact form that
	// would be sent if the ContentType property was
	// application/x-www-form-urlencoded). For example, if a request has two params:
	// param1="abc 123" and param2="abc-123", then GetUrlEncodedParams would return
	// "abc+123
	const char *getUrlEncodedParams(void);
	// Returns the request parameters in URL encoded form (i.e. in the exact form that
	// would be sent if the ContentType property was
	// application/x-www-form-urlencoded). For example, if a request has two params:
	// param1="abc 123" and param2="abc-123", then GetUrlEncodedParams would return
	// "abc+123
	const char *urlEncodedParams(void);


	// Uses the contents of the requestBody as the HTTP request body.
	bool LoadBodyFromBd(CkBinData &requestBody);


	// The HTTP protocol is such that all HTTP requests are MIME. For non-multipart
	// requests, this method may be called to set the MIME body of the HTTP request to
	// the exact contents of the byteData.
	// Note: A non-multipart HTTP request consists of (1) the HTTP start line, (2) MIME
	// header fields, and (3) the MIME body. This method sets the MIME body.
	bool LoadBodyFromBytes(CkByteData &byteData);


	// The HTTP protocol is such that all HTTP requests are MIME. For non-multipart
	// requests, this method may be called to set the MIME body of the HTTP request to
	// the exact contents of filePath.
	// Note: A non-multipart HTTP request consists of (1) the HTTP start line, (2) MIME
	// header fields, and (3) the MIME body. This method sets the MIME body.
	bool LoadBodyFromFile(const char *filePath);


	// Uses the contents of the requestBody as the HTTP request body. The charset indicates the
	// binary representation of the string, such as "utf-8", "utf-16", "iso-8859-*",
	// "windows-125*", etc. Any of the character encodings supported at the link below
	// are valid.
	bool LoadBodyFromSb(CkStringBuilder &requestBody, const char *charset);


	// The HTTP protocol is such that all HTTP requests are MIME. For non-multipart
	// requests, this method may be called to set the MIME body of the HTTP request to
	// the exact contents of bodyStr.
	// Note: A non-multipart HTTP request consists of (1) the HTTP start line, (2) MIME
	// header fields, and (3) the MIME body. This method sets the MIME body.
	// 
	// charset indicates the charset, such as "utf-8" or "iso-8859-1", to be used. The
	// HTTP body will contain the bodyStr converted to this character encoding.
	// 
	bool LoadBodyFromString(const char *bodyStr, const char *charset);


	// Removes all request parameters.
	void RemoveAllParams(void);


	// Removes all occurrences of a HTTP request header field. Always returns true.
	bool RemoveHeader(const char *name);


	// Removes a single HTTP request parameter by name.
	void RemoveParam(const char *name);


	// Parses a URL and sets the Path and query parameters (NumParams, GetParam,
	// GetParamName, GetParamValue).
	void SetFromUrl(const char *url);


	// Useful for sending HTTP requests where the body is a very large file. For
	// example, to send an XML HttpRequest containing a very large XML document, one
	// would set the HttpVerb = "POST", the ContentType = "text/xml", and then call
	// StreamBodyFromFile to indicate that the XML body of the request is to be
	// streamed directly from a file. When the HTTP request is actually sent, the body
	// is streamed directly from the file, and thus the file never needs to be loaded
	// in its entirety in memory.
	bool StreamBodyFromFile(const char *filePath);


	// This method is the same as StreamBodyFromFile, but allows for an offset and
	// number of bytes to be specified. The offset and numBytes are integers passed as
	// strings.
	bool StreamChunkFromFile(const char *path, const char *offset, const char *numBytes);


	// Makes the HttpRequest a GET request.
	// 
	// Important: This method is deprecated. An application should instead set the
	// HttpVerb property equal to "GET", and the ContentType equal to an empty string
	// (because GET requests have no request body).
	// 
	void UseGet(void);


	// Makes the HttpRequest a HEAD request.
	// 
	// Important: This method is deprecated. An application should instead set the
	// HttpVerb property equal to "HEAD", and the ContentType equal to an empty string
	// (because HEAD requests have no body).
	// 
	void UseHead(void);


	// Makes the HttpRequest a POST request that uses the
	// "application/x-www-form-urlencoded" content type.
	// 
	// Important: This method is deprecated. An application should instead set the
	// HttpVerb property equal to "POST", and the ContentType equal to
	// "application/x-www-form-urlencoded".
	// 
	void UsePost(void);


	// Makes the HttpRequest a POST request that uses the "multipart/form-data" content
	// type.
	// 
	// Important: This method is deprecated. An application should instead set the
	// HttpVerb property equal to "POST", and the ContentType equal to
	// "multipart/form-data".
	// 
	void UsePostMultipartForm(void);


	// Makes the HttpRequest a PUT request.
	// 
	// Important: This method is deprecated. An application should instead set the
	// HttpVerb property equal to "PUT", and the ContentType equal to
	// "application/x-www-form-urlencoded".
	// 
	void UsePut(void);


	// Makes the HttpRequest a POST request that uses the "multipart/form-data" content
	// type. To create a file upload request, call UseUpload and then call
	// AddFileForUpload for each file to be uploaded.
	// 
	// Important: This method is deprecated. An application should instead set the
	// HttpVerb property equal to "POST", and the ContentType equal to
	// "multipart/form-data".
	// 
	void UseUpload(void);


	// Makes the HttpRequest a PUT request that uses the "multipart/form-data" content
	// type. To create a file upload request (using the PUT verb), call UseUploadPut
	// and then call AddFileForUpload for each file to be uploaded.
	// 
	// Important: This method is deprecated. An application should instead set the
	// HttpVerb property equal to "PUT", and the ContentType equal to
	// "multipart/form-data".
	// 
	void UseUploadPut(void);


	// Makes the HttpRequest a POST request using the "application/xml" content type.
	// The request body is set to the XML string passed to this method.
	// 
	// Important: This method is deprecated. An application should instead set the
	// HttpVerb property equal to "POST", the ContentType equal to "text/xml", and the
	// request body should contain the XML document text.
	// 
	void UseXmlHttp(const char *xmlBody);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
