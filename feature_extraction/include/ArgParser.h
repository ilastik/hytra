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

    ArgParser(int argc, char** argv);
    void usage();
    void parse();
    bool hasKey(std::string key);
    void addRequiredArg(std::string key, Type type, std::string description = "");
    void addOptionalArg(std::string key, Type type, std::string description = "");

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
    Type findKey(std::string key);

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
