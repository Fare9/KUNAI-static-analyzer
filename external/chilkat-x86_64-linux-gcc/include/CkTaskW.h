// CkTaskW.h: interface for the CkTaskW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkTaskW_H
#define _CkTaskW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkWideCharBase.h"

class CkByteData;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkTaskW
class CK_VISIBLE_PUBLIC CkTaskW  : public CkWideCharBase
{
	

	private:
	
	// Don't allow assignment or copying these objects.
	CkTaskW(const CkTaskW &);
	CkTaskW &operator=(const CkTaskW &);

    public:
	CkTaskW(void);
	virtual ~CkTaskW(void);

	

	static CkTaskW *createNew(void);
	

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// true if the task status is "canceled", "aborted", or "completed". A task can
	// only reach the "canceled" status if it was activated via the Run method, made it
	// onto the internal thread pool thread's queue, was waiting for a pool thread to
	// become available, and was then canceled prior to the task actually starting.
	bool get_Finished(void);

	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort the Wait method. If
	// HeartbeatMs is 0 (the default), no AbortCheck event callbacks will fire. Note:
	// An asynchronous task running in a background thread (in one of the thread pool
	// threads) does not fire events. The task's event callbacks pertain only to the
	// Wait method.
	int get_HeartbeatMs(void);
	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort the Wait method. If
	// HeartbeatMs is 0 (the default), no AbortCheck event callbacks will fire. Note:
	// An asynchronous task running in a background thread (in one of the thread pool
	// threads) does not fire events. The task's event callbacks pertain only to the
	// Wait method.
	void put_HeartbeatMs(int newVal);

	// true if the task status is "empty" or "loaded". When a task is inert, it has
	// been loaded but is not scheduled to run yet.
	bool get_Inert(void);

	// Determines if the in-memory progress info event log is kept. The default value
	// is false and therefore no log is kept. To enable progress info logging, set
	// this property equal to true (prior to running the task).
	bool get_KeepProgressLog(void);
	// Determines if the in-memory progress info event log is kept. The default value
	// is false and therefore no log is kept. To enable progress info logging, set
	// this property equal to true (prior to running the task).
	void put_KeepProgressLog(bool newVal);

	// true if the task status is "queued" or "running". When a task is live, it is
	// either already running, or is on the thread pool thread's queue waiting for a
	// thread to become available.
	bool get_Live(void);

	// Indicates the percent completion while the task is running. The percent
	// completed information is only available in cases where it is possible to know
	// the percentage completed. For some methods, it is never possible to know, such
	// as for methods that establish TCP or TLS connections. For other methods it is
	// always possible to know -- such as for sending email (because the size of the
	// email to be sent is already known). For some methods, it may or may not be
	// possible to know the percent completed. For example, if an HTTP response is
	// "chunked", there is no Content-Length header and therefore the receiver has no
	// knowledge of the size of the forthcoming response body.
	// 
	// Also, the value of the PercentDoneScale property of the asynchronous method's
	// object determines the scale, such as 0 to 100, or 0 to 1000, etc.
	// 
	int get_PercentDone(void);

	// What would normally be a ProgressInfo event callback (assuming Chilkat supports
	// event callbacks for this language) is instead saved to an in-memory progress log
	// that can be examined and pruned while the task is still running. This property
	// returns the number of progress log entries that are currently available. (Note:
	// the progress log is only kept if the KeepProgressLog property is turned on. By
	// default, the KeepProgressLog is turned off.)
	int get_ProgressLogSize(void);

	// The LastErrorText for the task's asynchronous method.
	void get_ResultErrorText(CkString &str);
	// The LastErrorText for the task's asynchronous method.
	const wchar_t *resultErrorText(void);

	// Indicates the data type of the task's result. This property is only available
	// after the task has completed. Possible values are "bool", "int", "string",
	// "bytes", "object", and "void". For example, if the result data type is "bool",
	// then call GetResultBool to get the boolean result of the underlying asynchronous
	// method.
	// 
	// For example, if the synchronous version of the method returned a boolean, then
	// in the asynchronous version of the method, the boolean return value is made
	// available via the GetResultBool method.
	// 
	void get_ResultType(CkString &str);
	// Indicates the data type of the task's result. This property is only available
	// after the task has completed. Possible values are "bool", "int", "string",
	// "bytes", "object", and "void". For example, if the result data type is "bool",
	// then call GetResultBool to get the boolean result of the underlying asynchronous
	// method.
	// 
	// For example, if the synchronous version of the method returned a boolean, then
	// in the asynchronous version of the method, the boolean return value is made
	// available via the GetResultBool method.
	// 
	const wchar_t *resultType(void);

	// The current status of the task. Possible values are:
	//     "empty" -- The method call and arguments are not yet loaded into the task
	//     object. This can only happen if a task was explicitly created instead of being
	//     returned by a method ending in "Async".
	//     "loaded" -- The method call and arguments are loaded into the task object.
	//     "queued" -- The task is in the thread pool's queue of tasks awaiting to be
	//     run.
	//     "running" -- The task is currently running.
	//     "canceled" -- The task was canceled before it entered the "running" state.
	//     "aborted" -- The task was canceled while it was in the running state.
	//     "completed" -- The task completed. The success or failure depends on the
	//     semantics of the method call and the value of the result.
	void get_Status(CkString &str);
	// The current status of the task. Possible values are:
	//     "empty" -- The method call and arguments are not yet loaded into the task
	//     object. This can only happen if a task was explicitly created instead of being
	//     returned by a method ending in "Async".
	//     "loaded" -- The method call and arguments are loaded into the task object.
	//     "queued" -- The task is in the thread pool's queue of tasks awaiting to be
	//     run.
	//     "running" -- The task is currently running.
	//     "canceled" -- The task was canceled before it entered the "running" state.
	//     "aborted" -- The task was canceled while it was in the running state.
	//     "completed" -- The task completed. The success or failure depends on the
	//     semantics of the method call and the value of the result.
	const wchar_t *status(void);

	// The current status of the task as an integer value. Possible values are:
	//     1 -- The method call and arguments are not yet loaded into the task object.
	//     This can only happen if a task was explicitly created instead of being returned
	//     by a method ending in "Async".
	//     2 -- The method call and arguments are loaded into the task object.
	//     3 -- The task is in the thread pool's queue of tasks awaiting to be run.
	//     4 -- The task is currently running.
	//     5 -- The task was canceled before it entered the "running" state.
	//     6 -- The task was canceled while it was in the running state.
	//     7 -- The task completed. The success or failure depends on the semantics of
	//     the method call and the value of the result.
	int get_StatusInt(void);

	// A unique integer ID assigned to this task. The purpose of this property is to
	// help an application identify the task if a TaskCompleted event callback is used.
	int get_TaskId(void);

	// This is the value of the LastMethodSuccess property of the underlying task
	// object. This property is only valid for those methods where the
	// LastMethodSuccess property would be valid had the method been called
	// synchronously.
	// 
	// Important: This property is only meaningful for cases where the underlying
	// method call has a non-boolean return value (such as for methods that return
	// strings, other Chilkat objects, or integers). If the underlying method call
	// returns a boolean, then call the GetResultBool() method instead to get the
	// boolean return value.
	// 
	bool get_TaskSuccess(void);

	// An application may use this property to attach some user-specific information
	// with the task, which may be useful if a TaskCompleted event callback is used.
	void get_UserData(CkString &str);
	// An application may use this property to attach some user-specific information
	// with the task, which may be useful if a TaskCompleted event callback is used.
	const wchar_t *userData(void);
	// An application may use this property to attach some user-specific information
	// with the task, which may be useful if a TaskCompleted event callback is used.
	void put_UserData(const wchar_t *newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Marks an asynchronous task for cancellation. The expected behavior depends on
	// the current status of the task as described here:
	//     "loaded" - If the task has been loaded but has not yet been queued to run in
	//     the thread pool, then there is nothing to do. (There is nothing to cancel
	//     because the task's Run method has not yet been called.) The task will remain in
	//     the "loaded" state.
	//     "queued" - The task is marked for cancellation, is dequeued, and will not
	//     run. The task's status changes immediately to "canceled".
	//     "running" - The already-running task is marked for cancellation. The task's
	//     status will eventually change to "aborted" when the asynchronous method returns.
	//     At that point in time, the ResultErrorText property will contain the
	//     "LastErrorText" of the method call. In the case where a task is marked for
	//     cancellation just at the time it's completing, the task status may instead
	//     change to "completed".
	//     "canceled", "aborted", "completed" - In these cases the task has already
	//     finished, and there will be no change in status.
	// Cancel returns true if the task was in the "queued" or "running" state when it
	// was marked for cancellation. Cancel returns false if the task was in any other
	// state.
	// 
	// Important: Calling the Cancel method marks a task for cancellation. It sets a
	// flag in memory that the running task will soon notice and then abort. It is
	// important to realize that your application is likely calling Cancel from the
	// main UI thread, whereas the asynchronous task is running in a background thread.
	// If the task was in the "running" state when Cancel was called, it will still be
	// in the "running" state when Cancel returns. It will take a short amount of time
	// until the task actually aborts. This is because operating systems schedule
	// threads in time slices, and the thread needs one or more time slices to notice
	// the cancellation flag and abort. After calling Cancel, your application might
	// wish to call the Wait method to wait until the task has actually aborted, or it
	// could periodically check the task's status and then react once the status
	// changes to "aborted".
	// 
	bool Cancel(void);

	// Removes all entries from the progress info log.
	void ClearProgressLog(void);

	// Returns the binary bytes result of the task. The bytes are copied to the caller.
	bool CopyResultBytes(CkByteData &outBytes);

	// Returns the boolean result of the task.
	bool GetResultBool(void);

	// Returns the binary bytes result of the task. The bytes are transferred to the
	// caller, not copied. Call CopyResultBytes instead to copy the result bytes.
	bool GetResultBytes(CkByteData &outBytes);

	// Returns the integer result of the task.
	int GetResultInt(void);

	// Returns the string result of the task.
	bool GetResultString(CkString &outStr);
	// Returns the string result of the task.
	const wchar_t *getResultString(void);
	// Returns the string result of the task.
	const wchar_t *resultString(void);

	// Returns the name of the Nth progress info event logged. The 1st entry is at
	// index 0.
	bool ProgressInfoName(int index, CkString &outStr);
	// Returns the name of the Nth progress info event logged. The 1st entry is at
	// index 0.
	const wchar_t *progressInfoName(int index);

	// Returns the value of the Nth progress info event logged. The 1st entry is at
	// index 0.
	bool ProgressInfoValue(int index, CkString &outStr);
	// Returns the value of the Nth progress info event logged. The 1st entry is at
	// index 0.
	const wchar_t *progressInfoValue(int index);

	// Removes the Nth progress info log entry.
	void RemoveProgressInfo(int index);

	// If a taskCompleted callback function is passed in, then the task is started on
	// Node's internal thread pool. If no callback function is passed, then the task is
	// run synchronously. Queues the task to run on the internal Chilkat thread pool.
	bool Run(void);

	// Runs the task synchronously. Then this method returns after the task has been
	// run.
	bool RunSynchronously(void);

	// Convenience method to force the calling thread to sleep for a number of
	// milliseconds. (This does not cause the task's background thread to sleep.)
	void SleepMs(int numMs);

	// Waits for the task to complete. Returns when task has completed, or after maxWaitMs
	// milliseconds have elapsed. (A maxWaitMs value of 0 is to wait indefinitely.) Returns
	// (false) if the task has not yet been started by calling the Run method, or if
	// the maxWaitMs expired. If the task completed, was already completed, was canceled or
	// aborted, then this method returns true.
	bool Wait(int maxWaitMs);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
