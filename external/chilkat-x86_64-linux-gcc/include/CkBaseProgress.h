// CkBaseProgress.h: interface for the CkBaseProgress class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _CKBASEPROGRESS_H
#define _CKBASEPROGRESS_H

#include "CkObject.h"

#if !defined(BOOL_IS_TYPEDEF) && !defined(OBJC_BOOL_DEFINED)
#ifndef BOOL
#define BOOL int
#endif
#endif
	
#ifndef TRUE
#define TRUE 1
#endif
	
#ifndef FALSE
#define FALSE 0
#endif	
	
#if !defined(WIN32) && !defined(WINCE)
#include "SystemTime.h"              
#include "FileTime.h"                
#endif    
	
#include "ck_inttypes.h"

class CkTask;

#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 
// When creating an application class that inherits the CkBaseProgress base class, use the CK_BASEPROGRESS_API 
// definition to declare the overrides in the class header.  This has the effect that if for
// some unforeseen and unlikely reason the Chilkat event callback API changes, or if new
// callback methods are added in a future version, then you'll discover them at compile time
// after updating to the new Chilkat version.  
// For example:
/*
    class MyProgress : public CkBaseProgress
    {
	public:
	    CK_BASEPROGRESS_API

	...
    };
*/
#define CK_BASEPROGRESS_API \
    void AbortCheck(bool *abort);\
    void PercentDone(int pctDone, bool *abort);\
    void ProgressInfo(const char *name, const char *value);\
    void TextData(const char *data);\
    void BinaryData(const unsigned char *data, unsigned int numBytes);\
    void TaskCompleted(CkTask &task);

class CK_VISIBLE_PUBLIC CkBaseProgress : public CkObject 
{
    protected:
	void *m_impl;

	// No copy constructor or assignment allowed..
	CkBaseProgress(const CkBaseProgress &);
	CkBaseProgress &operator=(const CkBaseProgress &);

    public:
	CkBaseProgress();
	virtual ~CkBaseProgress();

	// This method is for internal use only.
	void *getProgressImpl(void);

	// Called periodically to check to see if the method call should be aborted.
	// The HeartbeatMs property controls the frequency of AbortCheck callbacks.
	virtual void AbortCheck(bool *abort) 
	    {
	    bool bAborted = AbortCheck();
	    if (abort) *abort = bAborted;
	    }
	// Return true if the method call should abort; return false for no abort.
	virtual bool AbortCheck(void) 
	    {
	    return false;
	    }

	// Called to indicate the current percentage completed for a method call.
	// PercentDone callbacks only happen where it makes sense and where it's possible.
	// Not all methods will have PercentDone callbacks.
	virtual void PercentDone(int pctDone, bool *abort) 
	    {
	    bool bAborted = PercentDone(pctDone);
	    if (abort) *abort = bAborted;
	    }
	// Return true if the method call should abort; return false for no abort.
	virtual bool PercentDone(int /*pctDone*/) 
	    { 
	    return false;
	    }

	// Open-ended callback where the name indicates the type of information provided.
	// The ProgressInfo callbacks depend on the method. 
	// To see what information is provided in ProgressInfo callbacks for any particular method,
	// if any, create a callback handler to log the callbacks for testing purposes.
	// Virtually all ProgressInfo callbacks should be self-explanatory.
	virtual void ProgressInfo(const char * /*name*/, const char * /*value*/) { }

	// Called when an asynchronous task completes, is aborted, canceled, etc.
	virtual void TaskCompleted(CkTask & /*task*/) { }

	// comment out the parameters names of TextData and BinaryData methods to stop unnecessary warnings being emitted about unused parameters
	virtual void TextData(const char *  /*data*/) { }
	virtual void BinaryData(const unsigned char * /*data*/, unsigned int /*numBytes*/) { }

};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


#endif
