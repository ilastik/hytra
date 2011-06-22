/*$Id: deArgParser.h 2029 2009-06-29 14:48:06Z tkroeger $*/

/*
 * deArgParser.h
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

#ifndef __ARGPARSER_H__
#define __ARGPARSER_H__

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstdlib>

class ArgParser
{
public:
    enum Type {Fractional, Integer, Integers, String, Bool, Strings, NoType};

    struct Key {
        Key() {}
        Key(std::string k, Type t, std::string desc = "")
                : key(k), type(t), description(desc), required(true) {}
        std::string key;
        Type type;
        std::string description;
        bool required;
    };

    ArgParser(int argc, char** argv)
    {
        argc_ = argc;
        argv_ = argv;
    }

    bool hasKey(std::string key)
    {
        std::map<std::string, int>::iterator pos = indexForKey_.find(key);
        return pos != indexForKey_.end();
    }

    void addRequiredArg(std::string key, Type type, std::string description)
    {
        args_[key] = Key(key, type, description);
        args_[key].required = true;
    }

    void addOptionalArg(std::string key, Type type, std::string description)
    {
        args_[key] = Key(key, type, description);
        args_[key].required = false;
    }


    void usage()
    {
        std::cerr << "usage: " << argv_[0] << std::endl;
        for (std::map<std::string, Key>::iterator it = args_.begin();
                it != args_.end(); ++it) {
            std::cerr << "  " << (!it->second.required ? "[" : "") << (*it).first << "=";
            switch (it->second.type) {
                case Fractional:
                    std::cerr << "double";
                    break;
                case Integer:
                    std::cerr << "integer";
                    break;
                case Integers:
                    std::cerr << "1,2,3,4,...";
                    break;
                case String:
                    std::cerr << "string";
                    break;
                case Bool:
                    std::cerr << "bool";
                case Strings:
                    std::cerr << "string1,string2,string3,...";
                default:
                    std::cerr << "unhandled type";
            }
            std::cerr << (!it->second.required ? "]" : "") << std::endl;
        }
    }

    void parse()
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
                if (t == NoType) {
                    //std::cerr << "key " << key << " is not allowed." << std::endl;
                    usage();
                    exit(-1);
                }
                else if (t == Integers) {
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
                else if (t == Integer) {
                    int i = atoi(value.c_str());
                    integerArgs_.push_back(i);
                    indexForKey_[key] = integerArgs_.size() - 1;
                }
                else if (t == Fractional) {
                    double f = atof(value.c_str());
                    fractionalArgs_.push_back(f);
                    indexForKey_[key] = fractionalArgs_.size() - 1;
                }
                else if (t == String) {
                    if (value[0] == '"') /*FIXME*/
                        stringArgs_.push_back(value.substr(1, value.size() - 2));
                    else
                        stringArgs_.push_back(value);
                    indexForKey_[key] = stringArgs_.size() - 1;
                }
                else if (t == Bool) {
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
                else if (t == Strings) {
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

    class Arg
    {
    public:
        Arg(unsigned int index,
                std::vector<int>* integerArgs,
                std::vector<double >* fractionalArgs,
                std::vector<std::vector<int> >* integersArgs,
                std::vector<std::string>* stringArgs,
                std::vector<bool>* boolArgs,
                std::vector<std::vector<std::string> >* stringsArgs)
                : index_(index),
                integerArgs_(integerArgs),
                fractionalArgs_(fractionalArgs),
                integersArgs_(integersArgs),
                stringArgs_(stringArgs),
                boolArgs_(boolArgs),
                stringsArgs_(stringsArgs) {}
        operator int () {
            return (*integerArgs_)[index_];
        }
        operator double () {
            return (*fractionalArgs_)[index_];
        }
        operator std::vector<int> () {
            return (*integersArgs_)[index_];
        }
        operator std::string () {
            return (*stringArgs_)[index_];
        }
        operator bool () {
            return (*boolArgs_)[index_];
        }
        operator std::vector<std::string> () {
            return (*stringsArgs_)[index_];
        }

    private:
        unsigned int                    index_;
        std::vector<int>*               integerArgs_;
        std::vector<double >*           fractionalArgs_;
        std::vector<std::vector<int> >* integersArgs_;
        std::vector<std::string>*       stringArgs_;
        std::vector<bool>*              boolArgs_;
        std::vector<std::vector<std::string> >* stringsArgs_;
    };

    Arg arg(std::string key) {
        if (!hasKey(key)) {
            //std::cerr << "trying to get passed argument value for key=" << key << " that was not passed." << std::endl;
            exit(1);
        }
        return Arg(indexForKey_[key], &integerArgs_, &fractionalArgs_, &integersArgs_, &stringArgs_, &boolArgs_, &stringsArgs_);
    }

private:
    Type findKey(std::string key) 
    {
        std::map<std::string, Key>::iterator pos = args_.find(key);
        if (pos == args_.end()) {
            return NoType;
        }
        else {
            return pos->second.type;
    }
}

    std::map<std::string, Key> args_;
    std::map<std::string, int>     indexForKey_;

    std::vector<double>            fractionalArgs_;
    std::vector<int>               integerArgs_;
    std::vector<std::vector<int> > integersArgs_;
    std::vector<std::string>       stringArgs_;
    std::vector<bool>              boolArgs_;
    std::vector<std::vector<std::string> > stringsArgs_;

    int                  argc_;
    char**               argv_;
};

#endif /*__ARGPARSER_H__*/
