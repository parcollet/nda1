// Copyright (c) 2019-2020 Simons Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "./clef.hpp"

namespace nda::clef::literals {

  constexpr auto i_ = placeholder<0>{};
  constexpr auto j_ = placeholder<1>{};
  constexpr auto k_ = placeholder<2>{};
  constexpr auto l_ = placeholder<3>{};

  constexpr auto bl_ = placeholder<4>{};

  constexpr auto w_ = placeholder<5>{};
  constexpr auto iw_ = placeholder<6>{};
  constexpr auto t_ = placeholder<7>{};
  constexpr auto tau_ = placeholder<8>{};

} // namespace nda::clef::literals
