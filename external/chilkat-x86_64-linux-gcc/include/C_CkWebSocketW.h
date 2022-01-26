// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkWebSocketWH
#define _C_CkWebSocketWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC void CkWebSocketW_setAbortCheck(HCkWebSocketW cHandle, BOOL (*fnAbortCheck)(void));
CK_C_VISIBLE_PUBLIC void CkWebSocketW_setPercentDone(HCkWebSocketW cHandle, BOOL (*fnPercentDone)(int pctDone));
CK_C_VISIBLE_PUBLIC void CkWebSocketW_setProgressInfo(HCkWebSocketW cHandle, void (*fnProgressInfo)(const wchar_t *name, const wchar_t *value));
CK_C_VISIBLE_PUBLIC void CkWebSocketW_setTaskCompleted(HCkWebSocketW cHandle, void (*fnTaskCompleted)(HCkTaskW hTask));

CK_C_VISIBLE_PUBLIC HCkWebSocketW CkWebSocketW_Create(void);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_Dispose(HCkWebSocketW handle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getCloseAutoRespond(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void  CkWebSocketW_putCloseAutoRespond(HCkWebSocketW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_getCloseReason(HCkWebSocketW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_closeReason(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getCloseReceived(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocketW_getCloseStatusCode(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_getDebugLogFilePath(HCkWebSocketW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC void  CkWebSocketW_putDebugLogFilePath(HCkWebSocketW cHandle, const wchar_t *newVal);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_debugLogFilePath(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getFinalFrame(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocketW_getFrameDataLen(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_getFrameOpcode(HCkWebSocketW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_frameOpcode(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocketW_getFrameOpcodeInt(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocketW_getIdleTimeoutMs(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void  CkWebSocketW_putIdleTimeoutMs(HCkWebSocketW cHandle, int newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getIsConnected(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_getLastErrorHtml(HCkWebSocketW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_lastErrorHtml(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_getLastErrorText(HCkWebSocketW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_lastErrorText(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_getLastErrorXml(HCkWebSocketW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_lastErrorXml(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getLastMethodSuccess(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void  CkWebSocketW_putLastMethodSuccess(HCkWebSocketW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getNeedSendPong(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getPingAutoRespond(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void  CkWebSocketW_putPingAutoRespond(HCkWebSocketW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getPongAutoConsume(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void  CkWebSocketW_putPongAutoConsume(HCkWebSocketW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getPongConsumed(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC int CkWebSocketW_getReadFrameFailReason(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_getUncommonOptions(HCkWebSocketW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_uncommonOptions(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_getVerboseLogging(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC void  CkWebSocketW_putVerboseLogging(HCkWebSocketW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkWebSocketW_getVersion(HCkWebSocketW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_version(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_AddClientHeaders(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_CloseConnection(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_GetFrameData(HCkWebSocketW cHandle, HCkString outStr);
CK_C_VISIBLE_PUBLIC const wchar_t *CkWebSocketW_getFrameData(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_GetFrameDataBd(HCkWebSocketW cHandle, HCkBinDataW binData);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_GetFrameDataSb(HCkWebSocketW cHandle, HCkStringBuilderW sb);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_LoadTaskCaller(HCkWebSocketW cHandle, HCkTaskW task);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_PollDataAvailable(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_ReadFrame(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkWebSocketW_ReadFrameAsync(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_SaveLastError(HCkWebSocketW cHandle, const wchar_t *path);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_SendClose(HCkWebSocketW cHandle, BOOL includeStatus, int statusCode, const wchar_t *reason);
CK_C_VISIBLE_PUBLIC HCkTaskW CkWebSocketW_SendCloseAsync(HCkWebSocketW cHandle, BOOL includeStatus, int statusCode, const wchar_t *reason);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_SendFrame(HCkWebSocketW cHandle, const wchar_t *stringToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC HCkTaskW CkWebSocketW_SendFrameAsync(HCkWebSocketW cHandle, const wchar_t *stringToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_SendFrameBd(HCkWebSocketW cHandle, HCkBinDataW bdToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC HCkTaskW CkWebSocketW_SendFrameBdAsync(HCkWebSocketW cHandle, HCkBinDataW bdToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_SendFrameSb(HCkWebSocketW cHandle, HCkStringBuilderW sbToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC HCkTaskW CkWebSocketW_SendFrameSbAsync(HCkWebSocketW cHandle, HCkStringBuilderW sbToSend, BOOL finalFrame);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_SendPing(HCkWebSocketW cHandle, const wchar_t *pingData);
CK_C_VISIBLE_PUBLIC HCkTaskW CkWebSocketW_SendPingAsync(HCkWebSocketW cHandle, const wchar_t *pingData);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_SendPong(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC HCkTaskW CkWebSocketW_SendPongAsync(HCkWebSocketW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_UseConnection(HCkWebSocketW cHandle, HCkRestW connection);
CK_C_VISIBLE_PUBLIC BOOL CkWebSocketW_ValidateServerHandshake(HCkWebSocketW cHandle);
#endif
