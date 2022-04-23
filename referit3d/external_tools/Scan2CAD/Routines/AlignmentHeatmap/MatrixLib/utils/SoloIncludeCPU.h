#pragma once

#include <Solo/utils/Common.h>
#include <Solo/utils/RuntimeAssertion.h>
#include <Solo/meta_structures/BasicTypes.h>
#include <Solo/meta_structures/SoA.h>
#include <Solo/meta_structures/TypeFinder.h>
#include <Solo/meta_structures/Dual.h>
#include <Solo/optimization_algorithms/SolverInterface.h>
#include <Solo/constraint_evaluation/AutoDiffCostFunction.h>
#include <Solo/constraint_evaluation/Bridge.h>

using solo::NullType;
using solo::EmptyType;
using solo::SoA;
using solo::SoASize;
using solo::SoAType;
using solo::SoAPointerList;
using solo::MemoryContainer;
using solo::MemoryTypeCPU;
using solo::MemoryTypeCUDA;
using solo::TL;
using solo::TypeAt;
using solo::I;
using solo::Unsigned2Type;
using solo::Int2Type;
using solo::Type2Type;
using solo::BaseType;
using solo::ResultType;
using solo::Param;
using solo::Params;
using solo::Constraint;
using solo::DataHolder;
using solo::Tuple;
using solo::makeTuple;
using solo::static_for;
using solo::Bridge;
using solo::real;
using solo::AddElements;