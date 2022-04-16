// This is a generated source file for Chilkat version 9.5.0.89
#ifndef _C_CkUrlWH
#define _C_CkUrlWH
#include "chilkatDefs.h"

#include "Chilkat_C.h"


CK_C_VISIBLE_PUBLIC HCkUrlW CkUrlW_Create(void);
CK_C_VISIBLE_PUBLIC void CkUrlW_Dispose(HCkUrlW handle);
CK_C_VISIBLE_PUBLIC void CkUrlW_getFrag(HCkUrlW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUrlW_frag(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC void CkUrlW_getHost(HCkUrlW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUrlW_host(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC void CkUrlW_getHostType(HCkUrlW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUrlW_hostType(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUrlW_getLastMethodSuccess(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC void  CkUrlW_putLastMethodSuccess(HCkUrlW cHandle, BOOL newVal);
CK_C_VISIBLE_PUBLIC void CkUrlW_getLogin(HCkUrlW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUrlW_login(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC void CkUrlW_getPassword(HCkUrlW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUrlW_password(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC void CkUrlW_getPath(HCkUrlW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUrlW_path(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC void CkUrlW_getPathWithQueryParams(HCkUrlW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUrlW_pathWithQueryParams(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC int CkUrlW_getPort(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC void CkUrlW_getQuery(HCkUrlW cHandle, HCkString retval);
CK_C_VISIBLE_PUBLIC const wchar_t *CkUrlW_query(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUrlW_getSsl(HCkUrlW cHandle);
CK_C_VISIBLE_PUBLIC BOOL CkUrlW_ParseUrl(HCkUrlW cHandle, const wchar_t *url);
#endif
