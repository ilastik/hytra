/*$Id: deArgParser.cpp 2033 2009-06-29 16:26:26Z tkroeger $*/

/*
 * deArgParser.cpp
 *
 * Copyright (c) 2008 Thorben Kroeger <thorben.kroeger@iwr.uni-heidelberg.de>
 *
 * This file is part of ms++.
 *
 * ms++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ms++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with ms++. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "ArgParser.h"

ArgParser::ArgParser(int argc, char** argv)
{
    argc_ = argc;
    argv_ = argv;
}

bool ArgParser::hasKey(std::string key)
{
    std::map<std::string, int>::iterator pos = indexForKey_.find(key);
    return pos != indexForKey_.end();
}

void ArgParser::addRequiredArg(std::string key, Type type, std::string description)
{
    args_[key] = Key(key, type, description);
    args_[key].required = true;
}

void ArgParser::addOptionalArg(std::string key, Type type, std::string description)
{
    args_[key] = Key(key, type, description);
    args_[key].required = false;
}

ArgParser::Type ArgParser::findKey(std::string key)
{
    std::map<std::string, Key>::iterator pos = args_.find(key);
    if (pos == args_.end()) {
        return ArgParser::NoType;
    }
    else {
        return pos->second.type;
    }
}

void ArgParser::usage()
{
    std::cerr << "usage: " << argv_[0] << std::endl;
    for (std::map<std::string, Key>::iterator it = args_.begin();
            it != args_.end(); ++it) {
        std::cerr << "  " << (!it->second.required ? "[" : "") << (*it).first << "=";
        switch (it->second.type) {
            case ArgParser::Fractional:
                std::cerr << "double";
                break;
            case ArgParser::Integer:
                std::cerr << "integer";
                break;
            case ArgParser::Integers:
                std::cerr << "1,2,3,4,...";
                break;
            case ArgParser::String:
                std::cerr << "string";
                break;
            case ArgParser::Bool:
                std::cerr << "bool";
            case ArgParser::Strings:
                std::cerr << "string1,string2,string3,...";
            default:
                std::cerr << "unhandled type";
        }
        std::cerr << (!it->second.required ? "]" : "") << std::endl;
    }
}

void ArgParser::parse()
{
    for (int i = 1; i < argc_; i++) {
        std::string token = argv_[i];
        std::string::size_type pos = token.find_first_of('=');
        if (pos == std::string::npos) {
            usage();
            exit(-1);
        }
        else {
            std::string key = token.substr(0, pos);
            std::string value = token.substr(pos + 1);

            Type t = findKey(key);
            if (t == ArgParser::NoType) {
                //std::cerr << "key " << key << " is not allowed." << std::endl;
                usage();
                exit(-1);
            }
            else if (t == ArgParser::Integers) {
                std::vector<int> v;
                std::string::size_type lastPos = value.find_first_not_of(',', 0);
                pos = value.find_first_of(',', lastPos);
                while (std::string::npos != pos || std::string::npos != lastPos) {
                    v.push_back(atoi(value.substr(lastPos, pos - lastPos).c_str()));
                    lastPos = value.find_first_not_of(',', pos);
                    pos = value.find_first_of(',', lastPos);
                }
                integersArgs_.push_back(v);
                indexForKey_[key] = integersArgs_.size() - 1;
            }
            else if (t == ArgParser::Integer) {
                int i = atoi(value.c_str());
                integerArgs_.push_back(i);
                indexForKey_[key] = integerArgs_.size() - 1;
            }
            else if (t == ArgParser::Fractional) {
                double f = atof(value.c_str());
                fractionalArgs_.push_back(f);
                indexForKey_[key] = fractionalArgs_.size() - 1;
            }
            else if (t == ArgParser::String) {
                if (value[0] == '"') /*FIXME*/
                    stringArgs_.push_back(value.substr(1, value.size() - 2));
                else
                    stringArgs_.push_back(value);
                indexForKey_[key] = stringArgs_.size() - 1;
            }
            else if (t == ArgParser::Bool) {
                bool b = false;
                if (value == "true")
                    b = true;
                else if (value == "false")
                    b = false;
                else {
                    //std::cerr << "key " << key << " has to be true of false" << std::endl;
                    usage();
                    exit(1);
                }
                boolArgs_.push_back(b);
                indexForKey_[key] = boolArgs_.size() - 1;
            }
            else if (t == ArgParser::Strings) {
                std::vector<std::string> v;
                std::string::size_type lastPos = value.find_first_not_of(',', 0);
                pos = value.find_first_of(',', lastPos);
                while (std::string::npos != pos || std::string::npos != lastPos) {
                    v.push_back(value.substr(lastPos, pos - lastPos).c_str());
                    lastPos = value.find_first_not_of(',', pos);
                    pos = value.find_first_of(',', lastPos);
                }
                stringsArgs_.push_back(v);
                indexForKey_[key] = stringsArgs_.size() - 1;
            }
        }
    }
    for (std::map<std::string, Key>::iterator it = args_.begin();
            it != args_.end(); ++it) {
        if (!hasKey((*it).first) && it->second.required) {
            //std::cerr << "required key \"" << it->first << "\" not present!" << std::endl;

            //std::cerr << std::endl << "Description of \"" << it->first << "\":" << std::endl << it->second.description << std::endl << std::endl;

            usage();
            exit(1);
        }
    }
}
