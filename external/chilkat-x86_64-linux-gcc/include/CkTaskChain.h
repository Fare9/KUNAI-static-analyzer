// CkTaskChain.h: interface for the CkTaskChain class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkTaskChain_H
#define _CkTaskChain_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkMultiByteBase.h"

class CkTask;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif


#undef Copy

// CLASS: CkTaskChain
class CK_VISIBLE_PUBLIC CkTaskChain  : public CkMultiByteBase
{
    private:

	// Don't allow assignment or copying these objects.
	CkTaskChain(const CkTaskChain &);
	CkTaskChain &operator=(const CkTaskChain &);

    public:
	CkTaskChain(void);
	virtual ~CkTaskChain(void);

	static CkTaskChain *createNew(void);
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	
		
	

	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// true if the task status is "canceled", "aborted", or "completed". A task chain
	// can only reach the "canceled" status if it was activated via the Run method,
	// made it onto the internal thread pool thread's queue, was waiting for a pool
	// thread to become available, and was then canceled prior to the task actually
	// starting.
	bool get_Finished(void);

	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort the Wait method. If
	// HeartbeatMs is 0 (the default), no AbortCheck event callbacks will fire. Note:
	// An asynchronous task chain running in a background thread (in one of the thread
	// pool threads) does not fire events. The task chain's event callbacks pertain
	// only to the Wait method.
	int get_HeartbeatMs(void);
	// The number of milliseconds between each AbortCheck event callback. The
	// AbortCheck callback allows an application to abort the Wait method. If
	// HeartbeatMs is 0 (the default), no AbortCheck event callbacks will fire. Note:
	// An asynchronous task chain running in a background thread (in one of the thread
	// pool threads) does not fire events. The task chain's event callbacks pertain
	// only to the Wait method.
	void put_HeartbeatMs(int newVal);

	// true if the task status is "empty" or "loaded". When a task chain is inert, it
	// has been loaded but is not scheduled to run yet.
	bool get_Inert(void);

	// true if the task status is "queued" or "running". When a task chain is live,
	// it is either already running, or is on the thread pool thread's queue waiting
	// for a thread to become available.
	bool get_Live(void);

	// The number of tasks contained within the task chain.
	int get_NumTasks(void);

	// The current status of the task chain. Possible values are:
	//     "empty" -- No tasks have yet been appended to the task chain.
	//     "loaded" -- The task chain has been loaded (appended) with one or more task
	//     objects.
	//     "queued" -- The task chain is in the thread pool's queue of tasks awaiting
	//     to be run.
	//     "running" -- The task chain is currently running.
	//     "canceled" -- The task chain was canceled before it entered the "running"
	//     state.
	//     "aborted" -- The task chain was canceled while it was in the running state.
	//     "completed" -- The task chain completed.
	void get_Status(CkString &str);
	// The current status of the task chain. Possible values are:
	//     "empty" -- No tasks have yet been appended to the task chain.
	//     "loaded" -- The task chain has been loaded (appended) with one or more task
	//     objects.
	//     "queued" -- The task chain is in the thread pool's queue of tasks awaiting
	//     to be run.
	//     "running" -- The task chain is currently running.
	//     "canceled" -- The task chain was canceled before it entered the "running"
	//     state.
	//     "aborted" -- The task chain was canceled while it was in the running state.
	//     "completed" -- The task chain completed.
	const char *status(void);

	// The current status of the task as an integer value. Possible values are:
	//     1 -- "empty" -- No tasks have yet been appended to the task chain.
	//     2 -- "loaded" -- The task chain has been loaded (appended) with one or more
	//     task objects.
	//     3 -- "queued" -- The task chain is in the thread pool's queue of tasks
	//     awaiting to be run.
	//     4 -- "running" -- The task chain is currently running.
	//     5 -- "canceled" -- The task chain was canceled before it entered the
	//     "running" state.
	//     6 -- "aborted" -- The task chain was canceled while it was in the running
	//     state.
	//     7 -- "completed" -- The task chain completed.
	int get_StatusInt(void);

	// If true then stops execution of the task chain if any individual task fails.
	// Task failure is defined by the standard LastMethodSuccess property. If false,
	// then all of the tasks in the chain will be run even if some fail. The default
	// value of this property is true.
	bool get_StopOnFailedTask(void);
	// If true then stops execution of the task chain if any individual task fails.
	// Task failure is defined by the standard LastMethodSuccess property. If false,
	// then all of the tasks in the chain will be run even if some fail. The default
	// value of this property is true.
	void put_StopOnFailedTask(bool newVal);



	// ----------------------
	// Methods
	// ----------------------
	// Appends a task to the task chain. Can fail if the task is already part of
	// another chain. (A task can only be part of a single chain.)
	bool Append(CkTask &task);


	// Cancels execution of the asynchronous task chain.
	bool Cancel(void);


	// Returns the Nth task in the chain. The 1st task is at index 0.
	// The caller is responsible for deleting the object returned by this method.
	CkTask *GetTask(int index);


	// If a taskCompleted callback function is passed in , then the task chain is
	// started on Node's internal thread pool. Each task in the chain will run, one
	// after the other. If no callback function is passed, the task chain runs
	// synchronously. Queues the task chain to run on the internal Chilkat thread pool.
	// Each task in the chain will run, one after the other.
	bool Run(void);


	// Runs the task chain synchronously. Then this method returns after all the tasks
	// in the chain have been run.
	bool RunSynchronously(void);


	// Convenience method to force the calling thread to sleep for a number of
	// milliseconds.
	void SleepMs(int numMs);


	// Waits for the task chain to complete. Returns when all of the tasks in the chain
	// have completed, or after maxWaitMs milliseconds have elapsed. (A maxWaitMs value of 0 is
	// to wait indefinitely.) Returns (false) if the task chain has not yet been
	// started by calling the Run method, or if the maxWaitMs expired. If the task chain
	// completed, was already completed, was canceled, or aborted, then this method
	// returns true.
	bool Wait(int maxWaitMs);






	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
