/****************************************************************************
** Meta object code from reading C++ file 'ActionRecognitionProject.h'
**
** Created: Thu Sep 6 10:45:47 2012
**      by: The Qt Meta Object Compiler version 63 (Qt 4.8.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../ActionRecognitionProject.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ActionRecognitionProject.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ActionRecognitionProject[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
       6,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      26,   25,   25,   25, 0x08,
      38,   25,   25,   25, 0x08,
      52,   25,   25,   25, 0x08,
      64,   25,   25,   25, 0x08,
      81,   25,   25,   25, 0x08,
     106,   25,   25,   25, 0x08,

       0        // eod
};

static const char qt_meta_stringdata_ActionRecognitionProject[] = {
    "ActionRecognitionProject\0\0loadFiles()\0"
    "playOrPause()\0nextFrame()\0loadMoSIFTFile()\0"
    "convertMoSIFTToMoFREAK()\0loadEverything()\0"
};

void ActionRecognitionProject::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        ActionRecognitionProject *_t = static_cast<ActionRecognitionProject *>(_o);
        switch (_id) {
        case 0: _t->loadFiles(); break;
        case 1: _t->playOrPause(); break;
        case 2: _t->nextFrame(); break;
        case 3: _t->loadMoSIFTFile(); break;
        case 4: _t->convertMoSIFTToMoFREAK(); break;
        case 5: _t->loadEverything(); break;
        default: ;
        }
    }
    Q_UNUSED(_a);
}

const QMetaObjectExtraData ActionRecognitionProject::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject ActionRecognitionProject::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_ActionRecognitionProject,
      qt_meta_data_ActionRecognitionProject, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ActionRecognitionProject::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ActionRecognitionProject::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ActionRecognitionProject::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ActionRecognitionProject))
        return static_cast<void*>(const_cast< ActionRecognitionProject*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int ActionRecognitionProject::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 6)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 6;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
