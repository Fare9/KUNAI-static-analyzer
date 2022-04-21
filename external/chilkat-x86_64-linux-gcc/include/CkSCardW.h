// CkSCardW.h: interface for the CkSCardW class.
//
//////////////////////////////////////////////////////////////////////

// This header is generated for Chilkat 9.5.0.89

#ifndef _CkSCardW_H
#define _CkSCardW_H
	
#include "chilkatDefs.h"

#include "CkString.h"
#include "CkClassWithCallbacksW.h"

class CkJsonObjectW;
class CkBinDataW;
class CkStringTableW;
class CkTaskW;
class CkBaseProgressW;



#if !defined(__sun__) && !defined(__sun)
#pragma pack (push, 8)
#endif
 

// CLASS: CkSCardW
class CK_VISIBLE_PUBLIC CkSCardW  : public CkClassWithCallbacksW
{
	private:
	bool m_cbOwned;

	private:
	
	// Don't allow assignment or copying these objects.
	CkSCardW(const CkSCardW &);
	CkSCardW &operator=(const CkSCardW &);

    public:
	CkSCardW(void);
	virtual ~CkSCardW(void);

	

	static CkSCardW *createNew(void);
	

	CkSCardW(bool bCallbackOwned);
	static CkSCardW *createNew(bool bCallbackOwned);

	
	void CK_VISIBLE_PRIVATE inject(void *impl);

	// May be called when finished with the object to free/dispose of any
	// internal resources held by the object. 
	void dispose(void);

	CkBaseProgressW *get_EventCallbackObject(void) const;
	void put_EventCallbackObject(CkBaseProgressW *progress);


	// BEGIN PUBLIC INTERFACE

	// ----------------------
	// Properties
	// ----------------------
	// The name of the active protocol if connected smart card reader, or an empty
	// string if not connected. Possible values are "T0", "T1", "raw", "undefined", or
	// "" if not connected to a reader.
	void get_ActiveProtocol(CkString &str);
	// The name of the active protocol if connected smart card reader, or an empty
	// string if not connected. Possible values are "T0", "T1", "raw", "undefined", or
	// "" if not connected to a reader.
	const wchar_t *activeProtocol(void);

	// This is the Current ATR of a card in the connected reader.
	void get_CardAtr(CkString &str);
	// This is the Current ATR of a card in the connected reader.
	const wchar_t *cardAtr(void);

	// The name of the currently connected smart card reader, or an empty string if not
	// connected.
	void get_ConnectedReader(CkString &str);
	// The name of the currently connected smart card reader, or an empty string if not
	// connected.
	const wchar_t *connectedReader(void);

	// Contains the string "user" or "system" if this object has established a context
	// (by calling EstablishContext). Contains the empty string if no context is
	// established.
	void get_Context(CkString &str);
	// Contains the string "user" or "system" if this object has established a context
	// (by calling EstablishContext). Contains the empty string if no context is
	// established.
	const wchar_t *context(void);

	// For Linux systems only. Specifies the full path of the libpcsclite.so shared
	// lib. This property should only be used if the libpcsclite.so is in a
	// non-standard location or if Chilkat cannot automatically located it.
	void get_PcscLibPath(CkString &str);
	// For Linux systems only. Specifies the full path of the libpcsclite.so shared
	// lib. This property should only be used if the libpcsclite.so is in a
	// non-standard location or if Chilkat cannot automatically located it.
	const wchar_t *pcscLibPath(void);
	// For Linux systems only. Specifies the full path of the libpcsclite.so shared
	// lib. This property should only be used if the libpcsclite.so is in a
	// non-standard location or if Chilkat cannot automatically located it.
	void put_PcscLibPath(const wchar_t *newVal);

	// The current status of the connected reader. Possible values are:
	//     "absent" - There is no card in the reader.
	//     "present" - There is a card in the reader, but it has not been moved into
	//     position for use.
	//     "swallowed" - There is a card in the reader in position for use. The card is
	//     not powered.
	//     "powered" - Power is being provided to the card, but the reader driver is
	//     unaware of the mode of the card.
	//     "negotiable" - The card has been reset and is awaiting PTS negotiation.
	//     "specific" - The card has been reset and specific communication protocols
	//     have been established.
	void get_ReaderStatus(CkString &str);
	// The current status of the connected reader. Possible values are:
	//     "absent" - There is no card in the reader.
	//     "present" - There is a card in the reader, but it has not been moved into
	//     position for use.
	//     "swallowed" - There is a card in the reader in position for use. The card is
	//     not powered.
	//     "powered" - Power is being provided to the card, but the reader driver is
	//     unaware of the mode of the card.
	//     "negotiable" - The card has been reset and is awaiting PTS negotiation.
	//     "specific" - The card has been reset and specific communication protocols
	//     have been established.
	const wchar_t *readerStatus(void);

	// The last error returned by an underlying PC/SC function. Can be one of the
	// following:
	//     "SCARDWREMOVED_CARD" - The smart card has been removed, so that further
	//     communication is not possible.
	//     "SCARDWRESET_CARD" - The smart card has been reset, so any shared state
	//     information is invalid.
	// ...
	void get_ScardError(CkString &str);
	// The last error returned by an underlying PC/SC function. Can be one of the
	// following:
	//     "SCARDWREMOVED_CARD" - The smart card has been removed, so that further
	//     communication is not possible.
	//     "SCARDWRESET_CARD" - The smart card has been reset, so any shared state
	//     information is invalid.
	// ...
	const wchar_t *scardError(void);



	// ----------------------
	// Methods
	// ----------------------
	// Establishes a temporary exclusive access mode for doing a series of commands in
	// a transaction.
	bool BeginTransaction(void);

	// Check the current status of the currently connected reader. Calling this method
	// updates the ReaderStatus, ActiveProtocol, and CardAtr properties. If this method
	// returns false, none of the properties are updated.
	bool CheckStatus(void);

	// Establish a connection to a reader. The reader is the name of a reader returned
	// from ListReaders. The shareMode can be "shared", "exclusive", or "direct". The preferredProtocol
	// can be "0" (valid only if the shareMode = "direct"), "T0", "T1", "raw", or
	// "no_preference". (No preference is effectively T0 or T1.)
	// 
	// If successful, the state of this object instance is that it's connected to the
	// reader.
	// 
	bool Connect(const wchar_t *reader, const wchar_t *shareMode, const wchar_t *preferredProtocol);

	// Terminates a connection with a reader. The disposition can be one of the following
	// values:
	//     "leave": Do nothing.
	//     "reset": Reset the card (warm reset).
	//     "unpower": Power down the card (cold reset).
	//     "eject": Eject the card.
	bool Disconnect(const wchar_t *disposition);

	// Ends a previously begun transaction. The disposition is the action to be taken on the
	// reader, and can be "leave" which is to do nothing, "reset", "unpower", or
	// "eject".
	bool EndTransaction(const wchar_t *disposition);

	// Creates an Application Context to the PC/SC Resource Manager. This must be the
	// first WinSCard function called in a PC/SC application. The scope can be "user" or
	// "system". After calling, this object will have context and all other methods
	// will use the established context. The "Context" property will hold the value
	// "user" or "system" if context was established, or will be empty if no context
	// was established.
	bool EstablishContext(const wchar_t *scope);

	// Returns JSON containing information about the smartcards currently inserted into
	// readers.
	bool FindSmartcards(CkJsonObjectW &json);

	// Get an attribute from the IFD Handler (reader driver).
	// 
	// The attr can be one of the following:
	//     "ASYNC_PROTOCOL_TYPES"
	//     "ATR_STRING"
	//     "CHANNEL_ID"
	//     "CHARACTERISTICS"
	//     "CURRENT_BWT"
	//     "CURRENT_CLK"
	//     "CURRENT_CWT"
	//     "CURRENT_D"
	//     "CURRENT_EBC_ENCODING"
	//     "CURRENT_F"
	//     "CURRENT_IFSC"
	//     "CURRENT_IFSD"
	//     "CURRENT_IO_STATE"
	//     "CURRENT_N"
	//     "CURRENT_PROTOCOL_TYPE"
	//     "CURRENT_W"
	//     "DEFAULT_CLK"
	//     "DEFAULT_DATA_RATE"
	//     "DEVICE_FRIENDLY_NAME"
	//     "DEVICE_IN_USE"
	//     "DEVICE_SYSTEM_NAME"
	//     "DEVICE_UNIT"
	//     "ESC_AUTHREQUEST"
	//     "ESC_CANCEL"
	//     "ESC_RESET"
	//     "EXTENDED_BWT"
	//     "ICC_INTERFACE_STATUS"
	//     "ICC_PRESENCE"
	//     "ICC_TYPE_PER_ATR"
	//     "MAX_CLK"
	//     "MAX_DATA_RATE"
	//     "MAX_IFSD"
	//     "MAXINPUT"
	//     "POWER_MGMT_SUPPORT"
	//     "SUPRESS_T1_IFS_REQUEST"
	//     "SYNC_PROTOCOL_TYPES"
	//     "USER_AUTH_INPUT_DEVICE"
	//     "USER_TO_CARD_AUTH_DEVICE"
	//     "VENDOR_IFD_SERIAL_NO"
	//     "VENDOR_IFD_TYPE"
	//     "VENDOR_IFD_VERSION"
	//     "VENDOR_NAME"
	// 
	// The attribute data is returned in bd.
	// 
	bool GetAttrib(const wchar_t *attr, CkBinDataW &bd);

	// Get a string typed attribute from the IFD Handler (reader driver).
	// 
	// The attr can be one of the following, but should be limited to the particular
	// attributes that return string values.
	//     "ASYNC_PROTOCOL_TYPES"
	//     "ATR_STRING"
	//     "CHANNEL_ID"
	//     "CHARACTERISTICS"
	//     "CURRENT_BWT"
	//     "CURRENT_CLK"
	//     "CURRENT_CWT"
	//     "CURRENT_D"
	//     "CURRENT_EBC_ENCODING"
	//     "CURRENT_F"
	//     "CURRENT_IFSC"
	//     "CURRENT_IFSD"
	//     "CURRENT_IO_STATE"
	//     "CURRENT_N"
	//     "CURRENT_PROTOCOL_TYPE"
	//     "CURRENT_W"
	//     "DEFAULT_CLK"
	//     "DEFAULT_DATA_RATE"
	//     "DEVICE_FRIENDLY_NAME"
	//     "DEVICE_IN_USE"
	//     "DEVICE_SYSTEM_NAME"
	//     "DEVICE_UNIT"
	//     "ESC_AUTHREQUEST"
	//     "ESC_CANCEL"
	//     "ESC_RESET"
	//     "EXTENDED_BWT"
	//     "ICC_INTERFACE_STATUS"
	//     "ICC_PRESENCE"
	//     "ICC_TYPE_PER_ATR"
	//     "MAX_CLK"
	//     "MAX_DATA_RATE"
	//     "MAX_IFSD"
	//     "MAXINPUT"
	//     "POWER_MGMT_SUPPORT"
	//     "SUPRESS_T1_IFS_REQUEST"
	//     "SYNC_PROTOCOL_TYPES"
	//     "USER_AUTH_INPUT_DEVICE"
	//     "USER_TO_CARD_AUTH_DEVICE"
	//     "VENDOR_IFD_SERIAL_NO"
	//     "VENDOR_IFD_TYPE"
	//     "VENDOR_IFD_VERSION"
	//     "VENDOR_NAME"
	// 
	bool GetAttribStr(const wchar_t *attr, CkString &outStr);
	// Get a string typed attribute from the IFD Handler (reader driver).
	// 
	// The attr can be one of the following, but should be limited to the particular
	// attributes that return string values.
	//     "ASYNC_PROTOCOL_TYPES"
	//     "ATR_STRING"
	//     "CHANNEL_ID"
	//     "CHARACTERISTICS"
	//     "CURRENT_BWT"
	//     "CURRENT_CLK"
	//     "CURRENT_CWT"
	//     "CURRENT_D"
	//     "CURRENT_EBC_ENCODING"
	//     "CURRENT_F"
	//     "CURRENT_IFSC"
	//     "CURRENT_IFSD"
	//     "CURRENT_IO_STATE"
	//     "CURRENT_N"
	//     "CURRENT_PROTOCOL_TYPE"
	//     "CURRENT_W"
	//     "DEFAULT_CLK"
	//     "DEFAULT_DATA_RATE"
	//     "DEVICE_FRIENDLY_NAME"
	//     "DEVICE_IN_USE"
	//     "DEVICE_SYSTEM_NAME"
	//     "DEVICE_UNIT"
	//     "ESC_AUTHREQUEST"
	//     "ESC_CANCEL"
	//     "ESC_RESET"
	//     "EXTENDED_BWT"
	//     "ICC_INTERFACE_STATUS"
	//     "ICC_PRESENCE"
	//     "ICC_TYPE_PER_ATR"
	//     "MAX_CLK"
	//     "MAX_DATA_RATE"
	//     "MAX_IFSD"
	//     "MAXINPUT"
	//     "POWER_MGMT_SUPPORT"
	//     "SUPRESS_T1_IFS_REQUEST"
	//     "SYNC_PROTOCOL_TYPES"
	//     "USER_AUTH_INPUT_DEVICE"
	//     "USER_TO_CARD_AUTH_DEVICE"
	//     "VENDOR_IFD_SERIAL_NO"
	//     "VENDOR_IFD_TYPE"
	//     "VENDOR_IFD_VERSION"
	//     "VENDOR_NAME"
	// 
	const wchar_t *getAttribStr(const wchar_t *attr);
	// Get a string typed attribute from the IFD Handler (reader driver).
	// 
	// The attr can be one of the following, but should be limited to the particular
	// attributes that return string values.
	//     "ASYNC_PROTOCOL_TYPES"
	//     "ATR_STRING"
	//     "CHANNEL_ID"
	//     "CHARACTERISTICS"
	//     "CURRENT_BWT"
	//     "CURRENT_CLK"
	//     "CURRENT_CWT"
	//     "CURRENT_D"
	//     "CURRENT_EBC_ENCODING"
	//     "CURRENT_F"
	//     "CURRENT_IFSC"
	//     "CURRENT_IFSD"
	//     "CURRENT_IO_STATE"
	//     "CURRENT_N"
	//     "CURRENT_PROTOCOL_TYPE"
	//     "CURRENT_W"
	//     "DEFAULT_CLK"
	//     "DEFAULT_DATA_RATE"
	//     "DEVICE_FRIENDLY_NAME"
	//     "DEVICE_IN_USE"
	//     "DEVICE_SYSTEM_NAME"
	//     "DEVICE_UNIT"
	//     "ESC_AUTHREQUEST"
	//     "ESC_CANCEL"
	//     "ESC_RESET"
	//     "EXTENDED_BWT"
	//     "ICC_INTERFACE_STATUS"
	//     "ICC_PRESENCE"
	//     "ICC_TYPE_PER_ATR"
	//     "MAX_CLK"
	//     "MAX_DATA_RATE"
	//     "MAX_IFSD"
	//     "MAXINPUT"
	//     "POWER_MGMT_SUPPORT"
	//     "SUPRESS_T1_IFS_REQUEST"
	//     "SYNC_PROTOCOL_TYPES"
	//     "USER_AUTH_INPUT_DEVICE"
	//     "USER_TO_CARD_AUTH_DEVICE"
	//     "VENDOR_IFD_SERIAL_NO"
	//     "VENDOR_IFD_TYPE"
	//     "VENDOR_IFD_VERSION"
	//     "VENDOR_NAME"
	// 
	const wchar_t *attribStr(const wchar_t *attr);

	// Get an unsigned integer typed attribute from the IFD Handler (reader driver).
	// 
	// The attr can be one of the following, but should be limited to the particular
	// attributes that return unsigned integer values.
	//     "ASYNC_PROTOCOL_TYPES"
	//     "ATR_STRING"
	//     "CHANNEL_ID"
	//     "CHARACTERISTICS"
	//     "CURRENT_BWT"
	//     "CURRENT_CLK"
	//     "CURRENT_CWT"
	//     "CURRENT_D"
	//     "CURRENT_EBC_ENCODING"
	//     "CURRENT_F"
	//     "CURRENT_IFSC"
	//     "CURRENT_IFSD"
	//     "CURRENT_IO_STATE"
	//     "CURRENT_N"
	//     "CURRENT_PROTOCOL_TYPE"
	//     "CURRENT_W"
	//     "DEFAULT_CLK"
	//     "DEFAULT_DATA_RATE"
	//     "DEVICE_FRIENDLY_NAME"
	//     "DEVICE_IN_USE"
	//     "DEVICE_SYSTEM_NAME"
	//     "DEVICE_UNIT"
	//     "ESC_AUTHREQUEST"
	//     "ESC_CANCEL"
	//     "ESC_RESET"
	//     "EXTENDED_BWT"
	//     "ICC_INTERFACE_STATUS"
	//     "ICC_PRESENCE"
	//     "ICC_TYPE_PER_ATR"
	//     "MAX_CLK"
	//     "MAX_DATA_RATE"
	//     "MAX_IFSD"
	//     "MAXINPUT"
	//     "POWER_MGMT_SUPPORT"
	//     "SUPRESS_T1_IFS_REQUEST"
	//     "SYNC_PROTOCOL_TYPES"
	//     "USER_AUTH_INPUT_DEVICE"
	//     "USER_TO_CARD_AUTH_DEVICE"
	//     "VENDOR_IFD_SERIAL_NO"
	//     "VENDOR_IFD_TYPE"
	//     "VENDOR_IFD_VERSION"
	//     "VENDOR_NAME"
	// 
	// Returns 0xFFFFFFFF on failure.
	// 
	unsigned long GetAttribUint(const wchar_t *attr);

	// Blocks execution until the current availability of the cards in a specific set
	// of readers changes.
	// 
	// This function receives a list of reader names in stReaderNames. It then blocks waiting
	// for a change in state to occur for a maximum blocking time of maxWaitMs (in
	// milliseconds) or forever if 0 is used.
	// 
	// Information about the current reader states and which reader(s) changed is
	// returned in json. See the example below for more information.
	// 
	// To wait for a reader event (reader added or removed) you may use the special
	// reader name "\\?PnP?\Notification".
	// 
	// To cancel the ongoing call, call Cancel().
	// 
	// The stReaderNames contains the reader names to check. The json is empty on input, and if
	// the call returns success contains information about the state (after the event
	// change) of each reader.
	// 
	bool GetStatusChange(int maxWaitMs, CkStringTableW &stReaderNames, CkJsonObjectW &json);

	// Creates an asynchronous task to call the GetStatusChange method with the
	// arguments provided. (Async methods are available starting in Chilkat v9.5.0.52.)
	// The caller is responsible for deleting the object returned by this method.
	CkTaskW *GetStatusChangeAsync(int maxWaitMs, CkStringTableW &stReaderNames, CkJsonObjectW &json);

	// Cancels an ongoing GetStatusChange method call. This would be called from a
	// separate thread in your application if GetStatusChange was called synchronously.
	bool GetStatusChangeCancel(void);

	// Returns a list of currently available reader groups on the system. The reader
	// groups are returned in readerGroups.
	bool ListReaderGroups(CkStringTableW &readerGroups);

	// Returns a list of currently available readers on the system.
	bool ListReaders(CkStringTableW &st);

	// Reestablishes a connection to a reader that was previously connected to using
	// Connect().
	// 
	// In a multi application environment it is possible for an application to reset
	// the card in shared mode. When this occurs any other application trying to access
	// certain commands will be returned the value SCARDWRESET_CARD. When this occurs
	// Reconnect() must be called in order to acknowledge that the card was reset and
	// allow it to change its state accordingly.
	// 
	// The shareMode can be "shared", "exclusive", or "direct". The preferredProtocol can be "0" (valid
	// only if the shareMode = "direct"), "T0", "T1", "raw", or "no_preference". (No
	// preference is effectively T0 or T1.) The action is the desired action taken on the
	// card/reader. It can be "leave", "reset", "unpower", or "eject".
	// 
	// If successful, the state of this object instance is that it's connected to the
	// reader.
	// 
	bool Reconnect(const wchar_t *shareMode, const wchar_t *preferredProtocol, const wchar_t *action);

	// Destroys a communication context to the PC/SC Resource Manager. This must be the
	// last function called in a PC/SC application.
	bool ReleaseContext(void);

	// Sends a command directly to the IFD Handler (reader driver) to be processed by
	// the reader.
	// 
	// This is useful for creating client side reader drivers for functions like PIN
	// pads, biometrics, or other extensions to the normal smart card reader that are
	// not normally handled by PC/SC.
	// 
	// The command data is sent in bdSend. The response is written to bdRecv.
	// 
	bool SendControl(unsigned long controlCode, CkBinDataW &bdSend, CkBinDataW &bdRecv);

	// Sends a command directly to the IFD Handler (reader driver) to be processed by
	// the reader.
	// 
	// This is useful for creating client side reader drivers for functions like PIN
	// pads, biometrics, or other extensions to the normal smart card reader that are
	// not normally handled by PC/SC.
	// 
	// The command data is provided as a hex string in sendData. The response is written to
	// bdRecv.
	// 
	bool SendControlHex(unsigned long controlCode, const wchar_t *sendData, CkBinDataW &bdRecv);

	// Sends an APDU to the smart card contained in the currently connected reader. The
	// protocol can be "T0", "T1", or "raw". The APDU to be sent is contained in bdSend. The
	// response from the card is contained in bdRecv. The maxRecvLen is the maximum response
	// size (in bytes) willing to be accepted.
	bool Transmit(const wchar_t *protocol, CkBinDataW &bdSend, CkBinDataW &bdRecv, int maxRecvLen);

	// Sends an APDU to the smart card contained in the currently connected reader. The
	// protocol can be "T0", "T1", or "raw". The APDU (in hexadecimal) to be sent is passed
	// in apduHex. The response from the card is contained in bdRecv. The maxRecvLen is the
	// maximum response size (in bytes) willing to be accepted.
	bool TransmitHex(const wchar_t *protocol, const wchar_t *apduHex, CkBinDataW &bdRecv, int maxRecvLen);





	// END PUBLIC INTERFACE


};
#if !defined(__sun__) && !defined(__sun)
#pragma pack (pop)
#endif


	
#endif
