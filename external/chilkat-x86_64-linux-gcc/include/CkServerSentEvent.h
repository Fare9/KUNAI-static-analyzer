// CkServerSentEvent.h: interface for the CkServerSentEvent class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkServerSentEvent_H
#define _CkServerSentEvent_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"




#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkServerSentEvent
class CK_VISIBLE_PUBLIC CkServerSentEvent  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkServerSentEvent(const CkServerSentEvent &);
	CkServerSentEvent &operator=(const CkServerSentEvent &);

    public:
	CkServerSentEvent(void);
	virtual ~CkServerSentEvent(void);

	static CkServerSentEvent *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The data for the server-side event. (If the "data" field was empty, then this
	// will be empty.)
	void get_Data(CkString &str);
	// The data for the server-side event. (If the "data" field was empty, then this
	// will be empty.)
	const char *data(void);

	// The name of the server-side event. (If the "event" field was not present, then
	// this will be empty.)
	void get_EventName(CkString &str);
	// The name of the server-side event. (If the "event" field was not present, then
	// this will be empty.)
	const char *eventName(void);

	// The content of the "id" field, if present.
	void get_LastEventId(CkString &str);
	// The content of the "id" field, if present.
	const char *lastEventId(void);

	// The integer value of the "retry" field, if present. Otherwise 0.
	int get_Retry(void);



	// ----------------------
	// Methods
	// ----------------------
	// Loads the multi-line event text into this object. For example, the eventText for a
	// Firebase event might look like this:
	// event: put
	// data: {"path": "/c", "data": {"foo": true, "bar": false}}
	bool LoadEvent(const char *eventText);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
