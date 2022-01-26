// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkWebSocket_H
#define _C_CkWebSocket_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkWebSocket_setAbortCheck(HCkWebSocket cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkWebSocket_setPercentDone(HCkWebSocket cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkWebSocket_setProgressInfo(HCkWebSocket cHandle, void (*fnProgressInfo)(const char *name, const char *value));
CK_C_VISIBLE_PUBLIC void CkWebSocket_setTaskCompleted(HCkWebSocket cHandle, void (*fnTaskCompleted)(HCkTask hTask));

CK_C_VISIBLE_PUBLIC void CkWebSocket_setAbortCheck2(HCkWebSocket cHandle, BOOL (*fnAbortCheck2)(void *pContext));
CK_C_VISIBLE_PUBLIC void CkWebSocket_setPercentDone2(HCkWebSocket cHandle, BOOL (*fnPercentDone2)(int pctDone, void *pContext));
CK_C_VISIBLE_PUBLIC void CkWebSocket_setProgressInfo2(HCkWebSocket cHandle, void (*fnProgressInfo2)(const char *name, const char *value, void *pContext));
CK_C_VISIBLE_PUBLIC void CkWebSocket_setTaskCompleted2(HCkWebSocket cHandle, void (*fnTaskCompleted2)(HCkTask hTask, void *pContext));

// setExternalProgress is for C callback functions defined in the external programming language (such as Go)
CK_C_VISIBLE_PUBLIC void CkWebSocket_setExternalProgress(HCkWebSocket cHandle, BOOL on);
CK_C_VISIBLE_PUBLIC void CkWebSocket_setCallbackContext(HCkWebSocket cHandle, void *pContext);

CK_C_VISIBLE_PUBLIC HCkWebSocket CkWebSocket_Create(void);
CK_C_VISIBLE_PUBLIC void CkWebSocket_Dispose(HCkWebSocket handle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getCloseAutoRespond(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_putCloseAutoRespond(HCkWebSocket cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkWebSocket_getCloseReason(HCkWebSocket cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_closeReason(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getCloseReceived(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocket_getCloseStatusCode(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_getDebugLogFilePath(HCkWebSocket cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void CkWebSocket_putDebugLogFilePath(HCkWebSocket cHandle, const char *newVal);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_debugLogFilePath(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getFinalFrame(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocket_getFrameDataLen(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_getFrameOpcode(HCkWebSocket cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_frameOpcode(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocket_getFrameOpcodeInt(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocket_getIdleTimeoutMs(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_putIdleTimeoutMs(HCkWebSocket cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getIsConnected(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_getLastErrorHtml(HCkWebSocket cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_lastErrorHtml(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_getLastErrorText(HCkWebSocket cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_lastErrorText(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_getLastErrorXml(HCkWebSocket cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_lastErrorXml(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getLastMethodSuccess(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_putLastMethodSuccess(HCkWebSocket cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getNeedSendPong(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getPingAutoRespond(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_putPingAutoRespond(HCkWebSocket cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getPongAutoConsume(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_putPongAutoConsume(HCkWebSocket cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getPongConsumed(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocket_getReadFrameFailReason(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_getUncommonOptions(HCkWebSocket cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_uncommonOptions(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getUtf8(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_putUtf8(HCkWebSocket cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_getVerboseLogging(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocket_putVerboseLogging(HCkWebSocket cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkWebSocket_getVersion(HCkWebSocket cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_version(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_AddClientHeaders(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_CloseConnection(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_GetFrameData(HCkWebSocket cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const char *CkWebSocket_getFrameData(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_GetFrameDataBd(HCkWebSocket cHandle, HCkBinData binData);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_GetFrameDataSb(HCkWebSocket cHandle, HCkStringBuilder sb);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_LoadTaskCaller(HCkWebSocket cHandle, HCkTask task);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_PollDataAvailable(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_ReadFrame(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC HCkTask CkWebSocket_ReadFrameAsync(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_SaveLastError(HCkWebSocket cHandle, const char *path);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_SendClose(HCkWebSocket cHandle, BOOL includeStatus, int statusCode, const char *reason);
CK_C_VISIBLE_PUBLIC HCkTask CkWebSocket_SendCloseAsync(HCkWebSocket cHandle, BOOL includeStatus, int statusCode, const char *reason);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_SendFrame(HCkWebSocket cHandle, const char *stringToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC HCkTask CkWebSocket_SendFrameAsync(HCkWebSocket cHandle, const char *stringToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_SendFrameBd(HCkWebSocket cHandle, HCkBinData bdToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC HCkTask CkWebSocket_SendFrameBdAsync(HCkWebSocket cHandle, HCkBinData bdToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_SendFrameSb(HCkWebSocket cHandle, HCkStringBuilder sbToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC HCkTask CkWebSocket_SendFrameSbAsync(HCkWebSocket cHandle, HCkStringBuilder sbToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_SendPing(HCkWebSocket cHandle, const char *pingData);
CK_C_VISIBLE_PUBLIC HCkTask CkWebSocket_SendPingAsync(HCkWebSocket cHandle, const char *pingData);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_SendPong(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC HCkTask CkWebSocket_SendPongAsync(HCkWebSocket cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_UseConnection(HCkWebSocket cHandle, HCkRest connection);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocket_ValidateServerHandshake(HCkWebSocket cHandle);
#endif
