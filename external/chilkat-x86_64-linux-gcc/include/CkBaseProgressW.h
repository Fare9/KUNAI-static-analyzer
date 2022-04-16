// CkBaseProgressW.h: interface for the CkBaseProgressW class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _CKBASEPROGRESSW_H
#define _CKBASEPROGRESSW_H

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
	
class CkTaskW;

#include "ck_inttypes.h"

#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 
// When creating an application class that inherits the CkBaseProgressW base class, use the CK_BASEPROGRESSW_API 
// definition to declare the overrides in the class header.  This has the effect that if for
// some unforeseen and unlikely reason the Chilkat event callback API changes, or if new
// callback methods are added in a future version, then you'll discover them at compile time
// after updating to the new Chilkat version.  
// For example:
/*
    class MyProgress : public CkBaseProgressW
    {
	public:
	    CK_BASEPROGRESSW_API

	...
    };
*/
#define CK_BASEPROGRESSW_API \
    void AbortCheck(bool *abort);\
    void PercentDone(int pctDone, bool *abort);\
    void ProgressInfo(const wchar_t *name, const wchar_t *value);\
    void TextData(const wchar_t *data);\
    void BinaryData(const unsigned char *data, unsigned int numBytes);\
    void TaskCompleted(CkTaskW &task);

class CK_VISIBLE_PUBLIC CkBaseProgressW : public CkObject 
{
    private:
	void *m_impl;

	// No copy constructor or assignment allowed..
	CkBaseProgressW(const CkBaseProgressW &);
	CkBaseProgressW &operator=(const CkBaseProgressW &);

    public:
	CkBaseProgressW();
	virtual ~CkBaseProgressW();

	// This method is for internal use only.
	void *getProgressImpl(void);

	// Called periodically to check to see if the method call should be aborted.
	// The HeartbeatMs property controls the frequency of AbortCheck callbacks.
	virtual void AbortCheck(bool *abort) 
	    { 
	    if (abort) *abort = false;
	    }

	// Called to indicate the current percentage completed for a method call.
	// PercentDone callbacks only happen where it makes sense and where it's possible.
	// Not all methods will have PercentDone callbacks.
	virtual void PercentDone(int /*pctDone*/, bool *abort) 
	    { 
	    if (abort) *abort = false;
	    }

	// Open-ended callback where the name indicates the type of information provided.
	// The ProgressInfo callbacks depend on the method. 
	// To see what information is provided in ProgressInfo callbacks for any particular method,
	// if any, create a callback handler to log the callbacks for testing purposes.
	// Virtually all ProgressInfo callbacks should be self-explanatory.
	virtual void ProgressInfo(const wchar_t * /*name*/, const wchar_t * /*value*/) { }

	// Called when an asynchronous task completes, is aborted, canceled, etc.
	virtual void TaskCompleted(CkTaskW & /*task*/) { }

	virtual void TextData(const wchar_t * /*data*/) { }
	virtual void BinaryData(const unsigned char * /*data*/, unsigned int /*numBytes*/) { }

};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


#endif
