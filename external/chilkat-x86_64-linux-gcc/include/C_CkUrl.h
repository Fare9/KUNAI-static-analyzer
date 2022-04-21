// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkUrl_H
#define _C_CkUrl_H
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkUrl CkUrl_Create(void);
CK_C_VISIBLE_PUBLIC void CkUrl_Dispose(HCkUrl handle);
CK_C_VISIBLE_PUBLIC void CkUrl_getFrag(HCkUrl cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUrl_frag(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC void CkUrl_getHost(HCkUrl cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUrl_host(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC void CkUrl_getHostType(HCkUrl cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUrl_hostType(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUrl_getLastMethodSuccess(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC void CkUrl_putLastMethodSuccess(HCkUrl cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkUrl_getLogin(HCkUrl cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUrl_login(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC void CkUrl_getPassword(HCkUrl cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUrl_password(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC void CkUrl_getPath(HCkUrl cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUrl_path(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC void CkUrl_getPathWithQueryParams(HCkUrl cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUrl_pathWithQueryParams(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC int CkUrl_getPort(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC void CkUrl_getQuery(HCkUrl cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const char *CkUrl_query(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUrl_getSsl(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUrl_getUtf8(HCkUrl cHandle);
CK_C_VISIBLE_PUBLIC void CkUrl_putUtf8(HCkUrl cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC BOOL CkUrl_ParseUrl(HCkUrl cHandle, const char *url);
#endif
